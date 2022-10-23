import sys

map_position_to_list_usage = {}
list_fn = ['runtime_ty_1.log', 'runtime_ty_2.log', 'runtime_ty_3.log', 'runtime_ty_4.log']
for fn in list_fn:
    with open(fn, 'r') as f:
        pos = 0
        for line in f:
            #received task ML
            if len(line.split('is received')) > 1:
                cons = [int(e) for e in line.split('It uses (')[1].split(')')[0].split(', ')]
                report = line.split('resource report is ')[1].split('\n')[0]
                time = int(report.split(', ')[5])/1000000 #in microseconds to seconds
                category = 1
                cons.append(time)
                cons = [category] + cons
                #task doesn't overconsume
                if int(line.split('res_exceeded variable is ')[1].split(',')[0]) == 0:
                    if pos not in map_position_to_list_usage:
                        map_position_to_list_usage[pos] = [cons]
                    else:
                        map_position_to_list_usage[pos].append(cons)
                    pos += 1
                else:
                    pass
            #receive task QC
            elif len(line.split('Resource report is')) > 1:
                category = -1
                task_id = int(line.split('tag ')[1])
                cons = [int(e) for e in line.split(', [')[1].split(', ')[:3]]
                alloc = [int(e) for e in line.split("'], [")[1].split(']')[0].split(', ')]
                report = line.split('Resource report is: ')[1].split(' with tag ')[0]
                time = int(report.split(', ')[5])/1000000 #in microseconds to seconds
                cons.append(time)
                alloc.append(time)
                resource_exceeded = 1 if cons[0] > alloc[0] or cons[1] > alloc[1] or cons[2] > alloc[2] else 0
                #task doesn't overconsume
                if not resource_exceeded:
                    cons = [category] + cons
                    if pos not in map_position_to_list_usage:
                        map_position_to_list_usage[pos] = [cons]
                    else:
                        map_position_to_list_usage[pos].append(cons)
                    pos += 1
                else:
                    pass

c = 0
for i in map_position_to_list_usage.keys():
    #print(f'key: {i}, num: {len(map_position_to_list_usage[i])}, value: {map_position_to_list_usage[i]}')
    #assert(len(map_position_to_list_usage[i]) == 3)
    #c = 0
    for l in map_position_to_list_usage[i]:
        category, cores, memory, disk, time = l
        if category == 1:
            print(f'{category}, {cores}, {memory}, {disk}, {time}')
            c += 1
        #if c == 30:
            #exit()
print(c)
