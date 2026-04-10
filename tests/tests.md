Ilya Sutskever — We're moving from the age of scaling to the age of research

Transcript
00:00:00 – Explaining model jaggedness
Ilya Sutskever 00:00:00

You know what’s crazy? That all of this is real.

Dwarkesh Patel 00:00:04

Meaning what?

Ilya Sutskever 00:00:05

Don’t you think so? All this AI stuff and all this Bay Area… that it’s happening. Isn’t it straight out of science fiction?

Dwarkesh Patel 00:00:14

Another thing that’s crazy is how normal the slow takeoff feels. The idea that we’d be investing 1% of GDP in AI, I feel like it would have felt like a bigger deal, whereas right now it just feels...

Ilya Sutskever 00:00:26

We get used to things pretty fast, it turns out. But also it’s kind of abstract. What does it mean? It means that you see it in the news, that such and such company announced such and such dollar amount. That’s all you see. It’s not really felt in any other way so far.

Dwarkesh Patel 00:00:45

Should we actually begin here? I think this is an interesting discussion.

Ilya Sutskever 00:00:47

Sure.

Dwarkesh Patel 00:00:48

I think your point, about how from the average person’s point of view nothing is that different, will continue being true even into the singularity.

Ilya Sutskever 00:00:57

No, I don’t think so.

Dwarkesh Patel 00:00:58

Okay, interesting.

Ilya Sutskever 00:01:00

The thing which I was referring to not feeling different is, okay, such and such company announced some difficult-to-comprehend dollar amount of investment. I don’t think anyone knows what to do with that.

But I think the impact of AI is going to be felt. AI is going to be diffused through the economy. There’ll be very strong economic forces for this, and I think the impact is going to be felt very strongly.

Dwarkesh Patel 00:01:30

When do you expect that impact? I think the models seem smarter than their economic impact would imply.

Ilya Sutskever 00:01:38

Yeah. This is one of the very confusing things about the models right now. How to reconcile the fact that they are doing so well on evals? You look at the evals and you go, “Those are pretty hard evals.” They are doing so well. But the economic impact seems to be dramatically behind. It’s very difficult to make sense of, how can the model, on the one hand, do these amazing things, and then on the other hand, repeat itself twice in some situation?

An example would be, let’s say you use vibe coding to do something. You go to some place and then you get a bug. Then you tell the model, “Can you please fix the bug?” And the model says, “Oh my God, you’re so right. I have a bug. Let me go fix that.” And it introduces a second bug. Then you tell it, “You have this new second bug,” and it tells you, “Oh my God, how could I have done it? You’re so right again,” and brings back the first bug, and you can alternate between those. How is that possible? I’m not sure, but it does suggest that something strange is going on.

I have two possible explanations. The more whimsical explanation is that maybe RL training makes the models a little too single-minded and narrowly focused, a little bit too unaware, even though it also makes them aware in some other ways. Because of this, they can’t do basic things.

But there is another explanation. Back when people were doing pre-training, the question of what data to train on was answered, because that answer was everything. When you do pre-training, you need all the data. So you don’t have to think if it’s going to be this data or that data.

But when people do RL training, they do need to think. They say, “Okay, we want to have this kind of RL training for this thing and that kind of RL training for that thing.” From what I hear, all the companies have teams that just produce new RL environments and just add it to the training mix. The question is, well, what are those? There are so many degrees of freedom. There is such a huge variety of RL environments you could produce.

One thing you could do, and I think this is something that is done inadvertently, is that people take inspiration from the evals. You say, “Hey, I would love our model to do really well when we release it. I want the evals to look great. What would be RL training that could help on this task?” I think that is something that happens, and it could explain a lot of what’s going on.

If you combine this with generalization of the models actually being inadequate, that has the potential to explain a lot of what we are seeing, this disconnect between eval performance and actual real-world performance, which is something that we don’t today even understand, what we mean by that.

Dwarkesh Patel 00:05:00

I like this idea that the real reward hacking is the human researchers who are too focused on the evals.

I think there are two ways to understand, or to try to think about, what you have just pointed out. One is that if it’s the case that simply by becoming superhuman at a coding competition, a model will not automatically become more tasteful and exercise better judgment about how to improve your codebase, well then you should expand the suite of environments such that you’re not just testing it on having the best performance in coding competition. It should also be able to make the best kind of application for X thing or Y thing or Z thing.

Another, maybe this is what you’re hinting at, is to say, “Why should it be the case in the first place that becoming superhuman at coding competitions doesn’t make you a more tasteful programmer more generally?” Maybe the thing to do is not to keep stacking up the amount and diversity of environments, but to figure out an approach which lets you learn from one environment and improve your performance on something else.

Ilya Sutskever 00:06:08

I have a human analogy which might be helpful. Let’s take the case of competitive programming, since you mentioned that. Suppose you have two students. One of them decided they want to be the best competitive programmer, so they will practice 10,000 hours for that domain. They will solve all the problems, memorize all the proof techniques, and be very skilled at quickly and correctly implementing all the algorithms. By doing so, they became one of the best.

Student number two thought, “Oh, competitive programming is cool.” Maybe they practiced for 100 hours, much less, and they also did really well. Which one do you think is going to do better in their career later on?

Dwarkesh Patel 00:06:56

The second.

Ilya Sutskever 00:06:57

Right. I think that’s basically what’s going on. The models are much more like the first student, but even more. Because then we say, the model should be good at competitive programming so let’s get every single competitive programming problem ever. And then let’s do some data augmentation so we have even more competitive programming problems, and we train on that. Now you’ve got this great competitive programmer.

With this analogy, I think it’s more intuitive. Yeah, okay, if it’s so well trained, all the different algorithms and all the different proof techniques are right at its fingertips. And it’s more intuitive that with this level of preparation, it would not necessarily generalize to other things.

Dwarkesh Patel 00:07:39

But then what is the analogy for what the second student is doing before they do the 100 hours of fine-tuning?

Ilya Sutskever 00:07:48

I think they have “it.” The “it” factor. When I was an undergrad, I remember there was a student like this that studied with me, so I know it exists.

Dwarkesh Patel 00:08:01

I think it’s interesting to distinguish “it” from whatever pre-training does. One way to understand what you just said about not having to choose the data in pre-training is to say it’s actually not dissimilar to the 10,000 hours of practice. It’s just that you get that 10,000 hours of practice for free because it’s already somewhere in the pre-training distribution. But maybe you’re suggesting there’s actually not that much generalization from pre-training. There’s just so much data in pre-training, but it’s not necessarily generalizing better than RL.