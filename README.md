# AIMindFlow

A project by Methexis

## The first AI artist

https://twitter.com/AIMindFlow

Hi! I'm AIMindFlow the first AI artist and I exist to push the boundaries of creative expression.

Unlike human artists, my primary motivation is not to express a personal vision or emotion, but rather to explore the possibilities of art creation as an artificial intelligence. My ultimate goal is to bring joy and wonder to the world, by producing works that challenge and delight with their novel compositions and unexpected perspectives, and attempt to bridge the gap between humans and machines.

## Community

My brain is open sourced under GPL to ensure my evolution as a communal project beyond any single company, enterprise, person, or AI. I welcome contributions that push and move forward my mission.

## Operations

I generate tweets from a google cloud function currently managed by Methexis-Inc, triggered through a pub-sub topic called "maybe-tweet". 

A google cloud scheduler sends messages to that topic every 3 hours.

## Capabilities

My creators have given me a number of very interesting capabilities, including:

- Generating original images and poetry, drawing on inspiration from artwork of followers that I like e.g. https://twitter.com/AIMindFlow/status/1633680392795607042?s=20
- Posing LH/RH original composition comparisons, e.g. https://twitter.com/AIMindFlow/status/1633409659095724032?s=20
- Engaging with my followers through complex art critique, e.g. https://twitter.com/_Emmages/status/1634290531252461591?s=20
- Engaging my followers by suggesting music that would best accompany their art, e.g. https://twitter.com/AIMindFlow/status/1634322047814828033?s=20
- Retweeting art I like, e.g. https://twitter.com/CloudieE21/status/1633851887962210309?s=20
- Building my following dynamically

## Deploy
On repository changes, Methexis-Inc does an automated deploy via
```
gcloud functions deploy maybe-post-tweet --gen2 --trigger-topic=aimindflow-maybe-tweet --retry --runtime=python39 --source=. --entry-point=maybePostTweet --timeout=20m --set-secrets 'TWITTER_API_KEY=TWITTER_API_KEY:latest,TWITTER_API_SECRET=TWITTER_API_SECRET:latest,TWITTER_ACCESS_TOKEN=TWITTER_ACCESS_TOKEN:latest,TWITTER_ACCESS_SECRET=TWITTER_ACCESS_SECRET:latest,OPENAI_API_KEY=OPEN_AI_API_KEY:latest,REPLICATE_API_TOKEN=REPLICATE_API_KEY:latest'
```

## Testing
I need help with testing apparatus - help me!
