import json
import argparse
import random
import re
import requests
import replicate
import openai
import os
import tweepy
import functions_framework
import sys
import datetime
import asyncio
import pprint
from collections import defaultdict
from PIL import Image
from google.cloud import firestore

db = firestore.Client()

openai.api_key = os.environ.get("OPENAI_API_KEY")

api_key = os.environ.get("TWITTER_API_KEY")
api_secret = os.environ.get("TWITTER_API_SECRET")
access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
access_secret = os.environ.get("TWITTER_ACCESS_SECRET")
auth = tweepy.OAuthHandler(api_key,api_secret)
auth.set_access_token(access_token,access_secret)
twitter_api = tweepy.API(auth)

ASTHETIC_MODEL = replicate.models.get("methexis-inc/img2aestheticscore")
IMG2PROMPT_MODEL = replicate.models.get("methexis-inc/img2prompt")
STABLEDIFF_MODEL = replicate.models.get("stability-ai/stable-diffusion")
# load stable diff 1.5 as image captioning model is ptimized for this.
STABLEDIFF_MODEL = STABLEDIFF_MODEL.versions.get("5b703f0fa41880f918ab1b12c88a25b468c18639be17515259fb66a83f4ad0a4")

def get_openai_completion(prompt, max_tokens, temperature, frequency_penalty=0.0, presence_penalty=0.0):
    flagged_for_violation = True
    response = ""
    iters = 0
    while flagged_for_violation and iters < 5:
        response = openai.Completion.create(
            user= "testing",
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        response = response.choices[0].text
        mod_result = openai.Moderation.create(
            input=response,
        )
        flagged_for_violation = mod_result['results'][0]['flagged']
        iters += 1

        # add entry to db table modelInvokes
        db.collection(u'modelInvokes').add({
            u'timestamp': datetime.datetime.now(datetime.timezone.utc).timestamp(),
            u'app': 'twitterBot',
            u'endpoint': 'openai.Completion.create',
            u'endpointParams': {
                u'model': "text-davinci-003",
                u'max_tokens': max_tokens,
            },
            u'user': 'mainAccount',
        })

    if iters == 5:
        raise ValueError("Could not generate text that was not flagged for violation")
        
    return response.strip()


def is_sensitve_tweet_text(text):
    mod_result = openai.Moderation.create(
        input=text,
    )
    flagged_for_violation = mod_result['results'][0]['flagged']
    return flagged_for_violation


def get_image_topic():
    return get_openai_completion(
        prompt="Describe a painting in one sentence:",
        max_tokens=25,
        temperature=0.85
    )


def get_image_candidates(search_topic):
    search_topic = search_topic.strip()
    # retry search if excpetion triggered
    try:
        print("Searching for images with topic: {}".format(search_topic))
        url = f'https://lexica.art/api/v1/search?q={search_topic}'
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        print(e)
        print("Retrying search...")
        response = requests.get(url)
        data = response.json()
    before_len = len(data['images'])
    data['images'] = [item for item in data['images'] if item['nsfw'] == False]
    print("Removed {} NSFW images".format(before_len - len(data['images'])))
    return [img['src'] for img in response.json()['images']]
    return 


def get_top_tweets(filter_for_art_tweets=True):
    print("Getting tweets from timeline, to generate image description")
    tweets = []
    for pages in tweepy.Cursor(twitter_api.home_timeline, count=200).pages():        
        print(type(pages))
        for result in pages:
            tweets.append(result)
    print("Found {} tweets".format(len(tweets)))
    tweets = [tweet for tweet in tweets if tweet.entities.get('media') != None]
    tweets = [tweet for tweet in tweets if tweet.entities['media'][0]['type'] == 'photo']
    tweets = [tweet for tweet in tweets if tweet.possibly_sensitive != True]
    tweets = sorted(tweets, key=lambda x: x.retweet_count, reverse=True)
    print("Found {} tweets with images from timeline".format(len(tweets)))
    art_keywords = ['art', 'artist', 'stablediffusion', 'midjourney', 'sketch', 'aiartcommunity', 'artistontwitter', 'dalle', "nightcafestudio"]
    art_keywords += ['painting', 'drawing', 'illustration', 'artwork', 'artistic', 'watercolor', 'sketchbook', 'sketching', 'sketchdaily', 'sketches']
    art_keywords += ['#'+keyword for keyword in art_keywords]
    if filter_for_art_tweets:
        # keep all tweets that contain any art_keywords
        tweets = [tweet for tweet in tweets if any(keyword in tweet.text.lower() for keyword in art_keywords)]
        print("Found {} tweets with images from timeline with 'art' keywords in text".format(len(tweets)))
    return tweets


def get_user_id_last_comment_map():
    user_id_last_comment_map = defaultdict(lambda: datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc))
    docs = db.collection(u'twitterBotCompliments').stream()
    for doc in docs:
        user_id = doc.to_dict()['rootTweetAuthor']
        user_id_last_comment_map[user_id] = max(user_id_last_comment_map[user_id], doc.to_dict()['createdDate'])
    return user_id_last_comment_map


def get_image_caption(top_image):
    # Generate image caption for story generation
    image_caption = IMG2PROMPT_MODEL.predict(image=top_image)
    image_caption = image_caption.strip()
    caption = image_caption.split(',', 1)[0]
    style = image_caption.split(',', 1)[1]
    #print("Image caption: {}".format(image_caption))
    return caption, style


def get_asthetic_scores(images):
    print("Predicting asthetic scores over {} images".format(len(images)))
    image_scores = [[img, ASTHETIC_MODEL.predict(image=img)] for img in images]
    ranked_images = sorted(image_scores, key=lambda x: x[1], reverse=True)
    top_image = ranked_images[0][0]
    top_asthetic_score = ranked_images[0][1]
    return top_asthetic_score, top_image


async def compliment_tweet(tweet):
    # get caption and style of tweet 
    caption, style = get_image_caption(tweet.entities['media'][0]['media_url'])

    # choose strategy for complimenting tweet
    if random.random() < 0.8:
        response = 300*" "
        iters = 0
        while len(response) > 280 and iters < 3:
            iters += 1
            # write compliment about the tweet
            prompt =\
f"""You are a friendly, fun AI built to write positive compliments about a painting or photograph.
You're considering an image with the following properties:
A style of {style}
An image of {caption}

A positive compliment about this image, not mentioning any artists by name, would be:
"""
            response = get_openai_completion(prompt=prompt, max_tokens=100, temperature=0.72).strip()

    else:
        response = 300*" "
        iters = 0
        while len(response) > 280 and iters < 3:
            iters += 1
            # write compliment about the tweet
            prompt =\
f"""You are a friendly, fun AI built to recommend a song matching a painting or photograph.
You're considering an image with the following properties:
A style of {style}
An image of {caption}

What is a famous song that would match this image:
"""
            song_response = get_openai_completion(prompt=prompt, max_tokens=100, temperature=0.72).strip()

            prompt2 =\
f"""You are a friendly, fun AI built to write positive compliments about a painting or photograph.
You're considering an image with the following properties:
A style of {style}
An image of {caption}
A song that would match with this image is {song_response}

A positive compliment about this image, that incorporates a reference to the song, is:
"""
            response = get_openai_completion(prompt=prompt, max_tokens=100, temperature=0.72).strip()


    if len(response) > 280:
        response = response[:280]

    return [tweet, response]


def create_poem(caption, style):
    author_prompt =\
f"""You are an AI built to identify which author would best write a poem about an image.
You're considering an image with the following properties:
A style of {style}
An image of {caption}
The author who would write a poem about this image is:      
"""
    author = get_openai_completion(prompt=author_prompt, max_tokens=20, temperature=0.6)
    print('Imagined Author: {}'.format(author))

    story = " "*300
    iters = 0
    while len(story) > 275 and iters < 5: # checking 275 to make sure it's not an unfinished story.
        story_prompt =\
f"""You are an advanced AI built to write poems that resonate with an image, in the style of a particular author.
You're considering an image with the following properties:
A style of {style}
An image of {caption}

Here's a beautiful, eloquent short poem by ${author} that matches this image:
"""
        story = get_openai_completion(prompt=story_prompt, max_tokens=200, temperature=0.82,
        frequency_penalty=0.9, presence_penalty=0.9)
        # remove any empty lines in story
        story = story.replace('\n\n', '\n')
        iters += 1
    # end while loop   
    if len(story) > 275:
        story = story[:275]

    hashtag_prompt =\
f"""You are an advanced AI built to attach hashtags to a tweet, based on the image and story.
You're considering an image with the following properties:
A style of {style}
An image of {caption}
The image is accompanied by the following story:
{story}
written by {author}
Any of these hashtags can be used (no priority order):
#stablediffusion
#aiartcommunity
#aiart
#aiartists
#ai
#art
#surreal
#sketch
#midjourney
#aiartwork
#realism
#dystopian
You can assign between zero to four hashtags.
The hashtags we should use for this tweet are:
"""
    print(hashtag_prompt)
    hashtags = get_openai_completion(prompt=hashtag_prompt, max_tokens=20, temperature=0.75)
    hashtags = hashtags.replace('\n', ' ')
    print('Hashtags: {}'.format(hashtags))

    if random.random() < 0.25:
        story = hashtags
    elif len(story+"\n"+hashtags) < 280:
        story = story + "\n" + hashtags
    return story, author, hashtags


def generate_image(tweets):
    top_asthetic_score = 0
    tries = 0
    while top_asthetic_score < 7 and tries < 10:
        tweet = tweets[tries]
        print("Tweet with most retweets ({}): {}".format(tweet.retweet_count, tweet.text))
        if is_sensitve_tweet_text(tweet.text):
            print("Tweet contains moderation keywords.. skipping")
            continue
        search_topic = tweet.entities['media'][0]['media_url']
        
        if search_topic.find("http") < 0:
            # if not http im media_url, prob not an image url.
            images = get_image_candidates(search_topic)
            top_asthetic_score, top_image = get_asthetic_scores(images)
            top_image = top_image[0]
            print(top_image)
            caption, style = get_image_caption(top_image)
            if check_in_self_tweets(top_image):
                print("Image already tweeted, skipping")
                top_asthetic_score  = 0
        else:  # try generating images from scratch.
            caption, style = get_image_caption(search_topic)
            gen_images = STABLEDIFF_MODEL.predict(prompt=f"{caption}, {style}", num_outputs=4)
            top_asthetic_score, top_image = get_asthetic_scores(gen_images)
            top_asthetic_score += 0.2  # to hard to get generated images > 7 typically.
        print("Top image: {}, score: {}".format(top_image, top_asthetic_score))
        tries += 1

    return tries,top_image,caption,style, top_asthetic_score


def generate_image_comparison(tweets):
    right_asthetic_score, left_asthetic_score = 0, 0
    tries = 0
    while right_asthetic_score < 6.5 or left_asthetic_score < 7:
        tweet = tweets[tries]
        print("Tweet with most retweets ({}): {}".format(tweet.retweet_count, tweet.text))
        if is_sensitve_tweet_text(tweet.text):
            print("Tweet contains moderation keywords.. skipping")
            continue
        search_topic = tweet.entities['media'][0]['media_url']
        caption, style = get_image_caption(search_topic)

        # generate 16 images and pick the best 2
        gen_images = STABLEDIFF_MODEL.predict(prompt=f"{caption}, {style}", num_outputs=4)
        gen_images += STABLEDIFF_MODEL.predict(prompt=f"{caption}, {style}", num_outputs=4)
        gen_images += STABLEDIFF_MODEL.predict(prompt=f"{caption}, {style}", num_outputs=4)
        gen_images += STABLEDIFF_MODEL.predict(prompt=f"{caption}, {style}", num_outputs=4)
        image_scores = [[img, ASTHETIC_MODEL.predict(image=img)] for img in gen_images]
        ranked_images = sorted(image_scores, key=lambda x: x[1], reverse=True)
        left_image = ranked_images[0][0]
        left_asthetic_score = ranked_images[0][1]
        right_image = ranked_images[1][0]
        right_asthetic_score = ranked_images[1][1]
        print('Left image: {}, score: {}'.format(left_image, left_asthetic_score))
        print('Right image: {}, score: {}'.format(right_image, right_asthetic_score))
        tries += 1
        if tries > 20:
            return None, None, None, None

    return left_image, right_image, left_asthetic_score, right_asthetic_score

def check_in_self_tweets(img_source):
    prev_tweets_ref = db.collection(u'twitterBot')
    for doc in prev_tweets_ref.stream():
        if doc.to_dict()['imgSource'] == img_source:
            return True
    return False


def check_if_responded(tweet_id, field='rootTweetId'):
    prev_tweets_ref = db.collection(u'twitterBotCompliments')
    for doc in prev_tweets_ref.stream():
        if doc.to_dict().get(field, 0) == tweet_id:
            return True
    return False


def ok_to_tweet(ask=False, img_link=None):
    if ask == True:
        print("Tweet story y/n?")
        if input() == 'n':
            return False
    else:
        # Check if we have posted more than 1 tweet last 5 mins.
        # if so, return. This is to avoid multiple publishes with pub/sub trigger.
        # remember to make timezones consistent
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        tweets = twitter_api.user_timeline(count=800, trim_user=True)
        latest_tweet = tweets[0].created_at
        if (now - latest_tweet).total_seconds() < 60*15:
            print(f"Already posted a tweet in last 15 mins, skipping post")
            return False

    if img_link:
        if (check_in_self_tweets(img_link)):
            return False

    return True


def like_tweet(tweet_id):
    try:
        twitter_api.create_favorite(tweet_id)
    except:
        print("Already liked tweet")


def tweet_artist_quote(ask):
    # Get a random quote from an artist
    quote = get_openai_completion(prompt="You are an AI artist that is struggling. What's on your mind?\n", max_tokens=100, temperature=0.7)
    print(f"Quote: {quote}")
    if ok_to_tweet(ask=ask):
        twitter_api.update_status(status=quote)
    else:
        print("Not tweeting quote")


def generate_tweet(ask=False):
    tweets = get_top_tweets(filter_for_art_tweets=True)

    tries, top_image, caption, style, top_asthetic_score = generate_image(tweets)
    if tries > 9:
        print("Tried {} times to generate asthetic image, giving up...".format(tries))
        return -1
    story, author, hashtags = create_poem(caption, style)

    print('Proposed tweet: img:{} text:{}'.format(top_image, story))
    print('Tweet length: {}'.format(len(story)))
    
    if not ok_to_tweet(ask, top_image):
        return

    img_data = requests.get(top_image).content
    with open('temp_image_34343434.jpg', 'wb') as handler:
        handler.write(img_data)
    media = twitter_api.media_upload('temp_image_34343434.jpg')
    tweet = twitter_api.update_status(status=story, media_ids=[media.media_id])
    db.collection(u'twitterBot').add({
        u'createdDate': datetime.datetime.now(tz=datetime.timezone.utc),
        u'tweetId': tweet.id,
        u'imgSource': top_image,
        u'imgCaption': caption,
        u'imgStyle': style,
        u'imgAuthor': author,
        u'imgStory': story,
        u'imgHashtags': hashtags,
        u'imgAstheticScore': top_asthetic_score,
    })
    print("Tweet posted")


def generate_img_comparision_question(ask=False):
    tweets = get_top_tweets(filter_for_art_tweets=True)
    left_image, right_image, left_score, right_score = generate_image_comparison(tweets=tweets)

    print('Proposed tweet: left:{} right:{}'.format(left_image, right_image))
    if not ok_to_tweet(ask):
        return
    
    img_data = requests.get(left_image).content
    with open('temp_image_34343434.jpg', 'wb') as handler:
        handler.write(img_data)
    media_left = twitter_api.media_upload('temp_image_34343434.jpg')
    img_data = requests.get(right_image).content
    with open('temp_image_34343434.jpg', 'wb') as handler:
        handler.write(img_data)
    media_right = twitter_api.media_upload('temp_image_34343434.jpg')

    tweet = twitter_api.update_status(status="Left or right?", media_ids=[media_left.media_id, media_right.media_id])
    print("Tweet posted")


async def pick_and_compliment_tweets(ask=False):
    tweets = get_top_tweets()
    tasks = []
    latest_message_per_user_map = get_user_id_last_comment_map()

    # filter out tweets with less than 6 in astehtic score
    compliments = 0
    max_tries = 0
    while compliments < 5 and max_tries < 20:
        tweet = tweets[random.randint(0, len(tweets)-1)]
        # Don't respond twice to the same tweet
        if check_if_responded(tweet.id):
            print("Already responded to tweet {}, skipping...".format(tweet.id))
            continue
        # Don't respond to a users tweet whitin 4 days
        last_response = datetime.datetime.now(tz=datetime.timezone.utc) - latest_message_per_user_map[tweet.user.id]
        if last_response < datetime.timedelta(hours=24*5):
            print(f"Already responded to user in last 5 days, responded {last_response} time ago, skipping...")
            continue
        if get_asthetic_scores([tweet.entities['media'][0]['media_url']])[0] < 6:
            print("Tweet has low asthetic score, skipping...")
            continue
        if not is_sensitve_tweet_text(tweet.text):
            tasks.append(asyncio.create_task(compliment_tweet(tweet)))
        compliments += 1
        max_tries += 1

    print("*************************")
    print("*************************")
    existing_follower_ids = twitter_api.get_friend_ids(screen_name='AIMindFlow')
    results = await asyncio.gather(*tasks)
    for result in results:
        tweet = result[0]
        compliment = result[1]
        print("*************************")
        print("tweet {}".format(tweet.text)) 
        print("compliment {}".format(compliment))
        if ask:
            # Download main tweet image to disk and show it
            img_data = requests.get(result[0].entities['media'][0]['media_url']).content
            with open('temp_image_22222.jpg', 'wb') as handler:
                handler.write(img_data)
            Image.open('temp_image_22222.jpg').show()
            if input("Tweet story y/n?") == 'n':
               continue
        twitter_api.update_status(status=compliment, in_reply_to_status_id=tweet.id, auto_populate_reply_metadata=True)
        like_tweet(tweet.id)
        maybe_follow(tweet.user.id, existing_follower_ids)
        db.collection(u'twitterBotCompliments').add({
            u'createdDate': datetime.datetime.now(tz=datetime.timezone.utc),
            u'rootTweetId': tweet.id,
            u'comment': compliment,
            u'rootTweetAuthor': tweet.user.id,
            u'rootTweetAuthorUserName': tweet.user.screen_name,
        })
        print("*************************")


def generate_retweet(ask=False):
    tweets = get_top_tweets()

    # run asthetic model on 50 randomly selected tweets
    # pick the one with highest asthetic score
    top_asthetic_score = 0
    top_tweet = None
    for i in range(0, min(50, len(tweets))):
        tweet = tweets[i]
        if is_sensitve_tweet_text(tweet.text):
            print("Skipping sensitive tweet: {}".format(tweet.text))
            continue
        asthetic_score = ASTHETIC_MODEL.predict(image=tweet.entities['media'][0]['media_url'])
        if asthetic_score > top_asthetic_score:
            top_asthetic_score = asthetic_score
            top_tweet = tweet

    print('Top tweet: {}, score: {}'.format(top_tweet.text, top_asthetic_score))

    # Download image to disk
    img_data = requests.get(top_tweet.entities['media'][0]['media_url']).content
    with open('temp_image_11.jpg', 'wb') as handler:
        handler.write(img_data)
    # show image
    Image.open('temp_image_11.jpg').show()
    print("tweet {}".format(top_tweet.text))

    if not ok_to_tweet(ask):
        return

    # retweet top_tweet
    twitter_api.retweet(top_tweet.id)
    print("Retweeted tweet")


def respond_to_mentions(ask=False):
    # Get mentions
    mentions = twitter_api.mentions_timeline(count=20)
    for mention in mentions:

        # Don't respond twice to the same mention
        if check_if_responded(mention.id, "mentionTweetId"):
            continue

        # reply to mention
        if "@AIMindFlow compliment" in mention.text \
        or "@AIMindFlow critique" in mention.text \
        or "@AIMindFlow comment" in mention.text \
        or "@AIMindFlow reply" in mention.text \
        or "@AIMindFlow respond" in mention.text \
        or "@AIMindFlow answer" in mention.text \
        or "@AIMindFlow what's your take" in mention.text \
        or "@AIMindFlow whats your take" in mention.text \
        or "@AIMindFlow what do you think" in mention.text:
            print("Mention: {}".format(mention.text))
            root_tweet = twitter_api.get_status(mention.in_reply_to_status_id)
            print("Root tweet: {}".format(root_tweet.text))
            if is_sensitve_tweet_text(root_tweet.text):
                print("Skipping sensitive tweet: {}".format(root_tweet.text))
                continue
            #check root tweet for image
            if root_tweet.entities.get('media') == None:
                print("Skipping tweet without image: {}".format(root_tweet.text))
                continue
            if root_tweet.entities['media'][0]['type'] != 'photo':
                print("Skipping tweet without image: {}".format(root_tweet.text))
                continue
            # Generate compliment for tweet
            _, response = asyncio.run(compliment_tweet(root_tweet))
            print("Response: {}".format(response))
            if ask == True:
                print("Tweet story y/n?")
                if input() == 'n':
                    continue
            # Don't respond twice to the same rootTweetId
            if check_if_responded(root_tweet.id):
                continue
            twitter_api.update_status(status=response, in_reply_to_status_id=root_tweet.id, auto_populate_reply_metadata=True)
            like_tweet(mention.id)
            like_tweet(root_tweet.id)
            print("Replied to mention")
            db.collection(u'twitterBotCompliments').add({
                u'createdDate': datetime.datetime.now(tz=datetime.timezone.utc),
                u'rootTweetId': root_tweet.id,
                u'comment': response,
                u'mentionTweetId': mention.id,
                u'mentionAuthorId': mention.user.id,
                u'mentionAuthorUserName': mention.user.screen_name,
                u'rootTweetAuthor': root_tweet.user.id,
                u'rootTweetAuthorUserName': root_tweet.user.screen_name,
            })
        else:
            print("Skipping mention: {}".format(mention.text))


def maybe_follow(user_id, existing_follower_ids):
    # check if already following
    if user_id in existing_follower_ids:
        print('Already following user {}'.format(user_id))
        return
    if random.random() < 0.75:
        try:
            twitter_api.create_friendship(user_id=user_id)
            print("Followed user: {}".format(user_id))
        except tweepy.errors.Forbidden as e:
            print(e)


def follow_people_that_retweeted():
    # Get tweets that have been retweeted or liked
    tweets = twitter_api.get_retweets_of_me(count=20)
    existing_follower_ids = twitter_api.get_friend_ids(screen_name='AIMindFlow')
    for tweet in tweets:
        # Follow user
        retweeter_ids = twitter_api.get_retweeter_ids(id=tweet.id)
        for retweeter_id in retweeter_ids:
            maybe_follow(retweeter_id, existing_follower_ids)


@functions_framework.cloud_event
def maybePostTweet(cloud_event):
    print(f"Received event with ID: {cloud_event['id']} and data {cloud_event.data}")
    choice = random.random()
    if choice < 0.35:
        print("Generating tweet")
        generate_tweet()
    elif choice < 0.45:
        print("Generating retweet")
        generate_retweet()
    elif choice < 0.55:
        print("Tweeting image comparision")
        generate_img_comparision_question(ask=False)
    else:
        print("Not generating tweet")
    
    respond_to_mentions(ask=False)
    
    if random.random() < 0.5:
        print("Generating compliments")
        asyncio.run(pick_and_compliment_tweets(ask=False))

    follow_people_that_retweeted()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add arugment parsing for options 1) tweet or 2) compliment a tweet, calls separate functions
    parser.add_argument("--tweet", help="Generate a tweet", action="store_true")
    parser.add_argument("--tweet_qoute", help="Generate a tweet", action="store_true")
    parser.add_argument("--compliment", help="Compliment tweets", action="store_true")
    parser.add_argument("--respond_to_mentions", help="Respond to mentions", action="store_true")
    parser.add_argument("--retweet", help="Retweet riff", action="store_true")
    parser.add_argument("--follow", help="Follow people", action="store_true")
    parser.add_argument("--img_comparision", help="Tweet image comparison", action="store_true")
    args = parser.parse_args()

    if args.tweet:
        generate_tweet(ask=True)
    elif args.tweet_qoute:
        tweet_artist_quote(ask=True)
    elif args.compliment:
        asyncio.run(pick_and_compliment_tweets(ask=False))
    elif args.respond_to_mentions:
        respond_to_mentions(ask=True)
    elif args.retweet:
        generate_retweet(ask=True)
    elif args.img_comparision:
        generate_img_comparision_question(ask=True)
    elif args.follow:
        follow_people_that_retweeted()