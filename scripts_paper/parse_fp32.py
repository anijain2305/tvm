import os
import sys

with open("tuned_perf.txt") as fh:
    lines = fh.read().split('\n')
    db = dict()
    for l in lines:
        if "Perf" in l:
            l = l.rstrip()
            entries = l.split('\t')
            if len(entries) > 1:
                framework, model, latency = entries[1], entries[2], float(entries[3])
                framework = 'FP32'
                if 'quantized' in model:
                    framework = 'Int8'
                    model = model.replace('-quantized-5batches-naive', '')
                # print(model, framework)
    
                if model not in db:
                    db[model] = dict()
                
                if framework not in db[model]:
                    db[model][framework] = list()
    
                db[model][framework].append(latency)


avg = lambda x : round(sum(x)/len(x), 6)
for model in db:
    for fw in db[model]:
        db[model][fw].append(avg(db[model][fw]))

print('', "FP32", "Int8", sep='\t')
for model in db:
    speedup = round(db[model]["FP32"][-1] / db[model]["Int8"][-1], 2)
    print(model, db[model]["FP32"][-1], db[model]["Int8"][-1], speedup, sep='\t')
print()
