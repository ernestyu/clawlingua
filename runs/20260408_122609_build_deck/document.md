# Ilya Sutskever — We're moving from the age of scaling to the age of research

Ilya Sutskever 00:01:38

Yeah. This is one of the very confusing things about the models right now. How to reconcile the fact that they are doing so well on evals? You look at the evals and you go, “Those are pretty hard evals.” They are doing so well. But the economic impact seems to be dramatically behind. It’s very difficult to make sense of, how can the model, on the one hand, do these amazing things, and then on the other hand, repeat itself twice in some situation?

An example would be, let’s say you use vibe coding to do something. You go to some place and then you get a bug. Then you tell the model, “Can you please fix the bug?” And the model says, “Oh my God, you’re so right. I have a bug. Let me go fix that.” And it introduces a second bug. Then you tell it, “You have this new second bug,” and it tells you, “Oh my God, how could I have done it? You’re so right again,” and brings back the first bug, and you can alternate between those. How is that possible? I’m not sure, but it does suggest that something strange is going on.

I have two possible explanations. The more whimsical explanation is that maybe RL training makes the models a little too single-minded and narrowly focused, a little bit too unaware, even though it also makes them aware in some other ways. Because of this, they can’t do basic things.

But there is another explanation. Back when people were doing pre-training, the question of what data to train on was answered, because that answer was everything. When you do pre-training, you need all the data. So you don’t have to think if it’s going to be this data or that data.