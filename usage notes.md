# note to sean

set effor / model to maximum before acting on this. 



# landing

we need a "see it before it happens" to reinforce the image showing two versions of the same woman. the hero should be captioned and managed not as a clockchain entry / timepoint render (it is, don't get confused) but as a marketing asset that shows the same woman as two characters in the moment. maybe after "AI-powered temporal simulations with causally coherent characters, dialog, and scenes — rendered from primary sources." we have a bold SEE YOURSELF AT THE PITCH BEFORE THE PITCH. 

is the clockchain data fresh? we need to update ever hour or something. 

the action items on Four ways to explore time have to convert not just to auth. do something then auth. like synthetic time travel let's give three free pre-auth renders to qualified visitors then require auth; flash generation only no pro; also allow these pre-auth users to browse the clockchain up to 10 entries then force auth. scaffold that into user tiers and setup stripe subscription tiers in addition to our credit system. 

Recent moments from the Clockchain image loading is failing. 

every image on https://timepointai.com/blog is failing. 

(wtf why did playwright miss all these dead images?) 


loading speed on e.g. https://app.timepointai.com/moment/-42/october/23/0900/macedonia-modern-day-greece/philippi/philippi/battle-of-philippi-42-greece-philippi-488d58 is very bad like 30 seconds. is that a railwy config thing? our stack? it feels like an error it's so slow. this cannot be allowed and should have been caught by playwright and dashboard latency management. 

do we need to cdn the images? to some extent we could roll our own CDV on railway or at least our own media server? down to use a third party too. 

on timepoint pages "render this moment" is a bad conversion. we have assets not displaying like dialog, and features like time travel fw and backward and chat w characters; those should be the conversions. capture users with "chat w this character" and then gate how many turns until auth they can chat. 


an unauthed user on eg. https://app.timepointai.com/moment/-42/october/23/0900/macedonia-modern-day-greece/philippi/philippi/battle-of-philippi-42-greece-philippi-488d58 will see the full left nav; that should only be if authenticated maybe grey out Pro, Billing, Settings if unauth? or hide whole thing? 

we don't have good use case conversions on dashboard, generate, or pro. we need much more solid conversion use cases. like "Simulate a Board Meeting before It Happens" kind of thing. we need leaders and hints and invocation pills. almost like but not quite like: Vertical Application Templates, Industry-Specific Accelerators, or No-Code/Low-Code Use Case Templates. 

it still says "twitter" in someplaces but it's X now fix all. 

https://app.timepointai.com/clockchain should allow free infinite browsing of gallery view. https://app.timepointai.com/clockchain needs a different mechanism for the long scroll of history. what if we did something kind wild like a curtain effect / keystone collapse into lower / higher resolution so you were almost like lensing the current moment out to bigger thumbs? I don't want bogged down js but there may be a way to do that w/ some clever image rendering / image resizing approaches. what is the most simple way to have a better interface? we could also do a timeline on the bottom like a pagination or gallery component slider so you see a range of time wherever you click on the time spectrum slider; or perhaps we are more playful with the time travel part and make it more like a synthwave-meets-back-to-the-future subtle little play with a time counter and year dashboard? right now you have to scroll swipe like 20 times to get to the bottom, unacceptable ux/ui. 

I need a developer candy option for my autistic need friends like 1000 mcp credits and a preauth through a link or something and then they oauth on top? how might I send my nerd friends a single link / key and they can do cool shit immediately no friction? a single point of friction fucks it all up. 

should we pypi? 

I also need investor candy and it's simple: I want a robust profile of timepoint as a company and me as an entpreneur (I can provide a markdown about me) and have a simple semantic clean url w fund name that I put in in my Candy generator admin dashboard page where I can choose investor candy and it gives me a url where my pitch meeting with the investment team is already built. this means entities and entity grounding; dialog; visualization with grounding. we should also consider visual precedent based on llm driven search discovery of images of the real people and try to ground the simulated image against it. in this case they have to auth and get the normal number free credits but they can see this private meeting once they auth. 

finally I need sales candy for law firms and funds and PE firms and private investors and entrepreneurs and startups. similar idea of a candy generator that makes a simulation of the meeting in advance. perhaps this matches / leverages / peers skipmeetings? 

the app dashboard should not be about rendering history like right now it encourages simulating the fall of Rome. we want to focus on forward-facing simualtions with the clockchain the past. the front of the lightgone is generate and pro; the back of the lightcone is clockchain. 

is search in a proper user silo and handling clockchain appropriately? 

some clockchain events load at > 1 minute to render in the browers. wildly unacceptable. and then some e.g. https://app.timepointai.com/moment/0/january/1/1200/unknown/unknown/unknown/flight-of-thomas-palaiologos-to-rome-1460-italy-rome-c8778f have no images? what's failing? how did playwright possibly miss this? 

RELATED MOMENTS needs to match the other time interface we choose on not be a rote list; should have thumbs. 

where do devs manage api keys? mcp keys? we need that section. 

why isn't the blog in the nav? 

why doesn't landing cover all our products? give each a section. make sure Dev covers all too. on / dev remove the image and the Causal decision graph. move "Causal decision graph" to the landing page as its own component. move "What your agent helps you see" to landing as well as its own component. convert devs specifically to the api / mcp management page we build. same credit schema. in Developers list more advantages of the stack in terms of its features, maybe some tables of features / advantages across the various component apps. 

what do we need to do to fix google oauth? apple? 

does timepoint-api-gateway have duplicate railway services? I see two containers but only one connects to postgres; fix and name appropriately. 






