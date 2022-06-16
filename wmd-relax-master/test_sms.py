from sentence_mover import SentenceMoverSim
from icecream import ic
from time import perf_counter

sms = SentenceMoverSim()


ref = ["knee injuries limit your motion", "sitting or standing doesn't require motion"]
perfect = ["knee injuries limit your motion", "sitting or standing doesn't require motion"]
good = ["knee injuries how much you can move", "you don't need to move in order to sit or stand"]
on_topic_bad = ["she sprained her knee", "you can injure knee by forgetting to stretch first"]
off_topic_bad = ["I am losing the will to live"]

ref1 = ["higher education requires tuition"]
good1 = ["school charge tuition fees"]
good2 = ["schools have fees", "poor people cannot afford school"]
bad1 = ["this sentence is irrelevant"]

start = perf_counter()
ic(sms.compute(ref, perfect))
ic(sms.compute(ref, good))
ic(sms.compute(ref, on_topic_bad))
ic(sms.compute(ref, off_topic_bad))

# one-to-one comparison
ic(sms.compute(ref1, good1))
ic(sms.compute(ref1, good2))
ic(sms.compute(ref1, bad1))
print(f"finished singles in {perf_counter() - start}s")

start = perf_counter()
ic(sms.batch_compute(
	[ref, ref, ref, ref, ref1, ref1, ref1], 
	[perfect, good, on_topic_bad, off_topic_bad, good1, good2, bad1],
	7
))
print(f"finished batch in {perf_counter() - start}s")
