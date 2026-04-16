Ilya Sutskever — We're moving from the age of scaling to the age of research
– Explaining model jaggedness
Ilya Sutskever
You know what’s crazy? That all of this is real.
Dwarkesh Patel
Meaning what?
Ilya Sutskever
Don’t you think so? All this AI stuff and all this Bay Area… that it’s happening. Isn’t it straight out of science fiction?
Dwarkesh Patel
Another thing that’s crazy is how normal the slow takeoff feels. The idea that we’d be investing 1% of GDP in AI, I feel like it would have felt like a bigger deal, whereas right now it just feels...
Ilya Sutskever
We get used to things pretty fast, it turns out. But also it’s kind of abstract. What does it mean? It means that you see it in the news, that such and such company announced such and such dollar amount. That’s all you see. It’s not really felt in any other way so far.
Dwarkesh Patel
Should we actually begin here? I think this is an interesting discussion.
Ilya Sutskever
Sure.
Dwarkesh Patel
I think your point, about how from the average person’s point of view nothing is that different, will continue being true even into the singularity.
Ilya Sutskever
No, I don’t think so.
Dwarkesh Patel
Okay, interesting.
Ilya Sutskever
The thing which I was referring to not feeling different is, okay, such and such company announced some difficult-to-comprehend dollar amount of investment. I don’t think anyone knows what to do with that.
But I think the impact of AI is going to be felt. AI is going to be diffused through the economy. There’ll be very strong economic forces for this, and I think the impact is going to be felt very strongly.
Dwarkesh Patel
When do you expect that impact? I think the models seem smarter than their economic impact would imply.
Ilya Sutskever
Yeah. This is one of the very confusing things about the models right now. How to reconcile the fact that they are doing so well on evals? You look at the evals and you go, “Those are pretty hard evals.” They are doing so well. But the economic impact seems to be dramatically behind. It’s very difficult to make sense of, how can the model, on the one hand, do these amazing things, and then on the other hand, repeat itself twice in some situation?
An example would be, let’s say you use vibe coding to do something. You go to some place and then you get a bug. Then you tell the model, “Can you please fix the bug?” And the model says, “Oh my God, you’re so right. I have a bug. Let me go fix that.” And it introduces a second bug. Then you tell it, “You have this new second bug,” and it tells you, “Oh my God, how could I have done it? You’re so right again,” and brings back the first bug, and you can alternate between those. How is that possible? I’m not sure, but it does suggest that something strange is going on.
I have two possible explanations. The more whimsical explanation is that maybe RL training makes the models a little too single-minded and narrowly focused, a little bit too unaware, even though it also makes them aware in some other ways. Because of this, they can’t do basic things.
But there is another explanation. Back when people were doing pre-training, the question of what data to train on was answered, because that answer was everything. When you do pre-training, you need all the data. So you don’t have to think if it’s going to be this data or that data.
But when people do RL training, they do need to think. They say, “Okay, we want to have this kind of RL training for this thing and that kind of RL training for that thing.” From what I hear, all the companies have teams that just produce new RL environments and just add it to the training mix. The question is, well, what are those? There are so many degrees of freedom. There is such a huge variety of RL environments you could produce.
One thing you could do, and I think this is somethin