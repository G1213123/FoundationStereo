import json
def print_m(j, name):
    print(name)
    metrics = j.get('metrics', [])
    for i, m in enumerate(metrics):
        v = m.get('values', [])
        print(f'  metric {i} values len: {len(v)}')
        if v and 'instances' in v[0]:
            print(f'    instances: {len(v[0][""instances""])}')
        if v and 'values' in v[0]:
            print(f'    inner values: {len(v[0][""values""])}')

with open(r'C:\Users\1213123\Documents\Scripts\FoundationStereo\scripts\run_files\batch_inputs\seq37\step0.frame_data.json') as f: print_m(json.load(f), 'batch_inputs')
with open(r'C:\Users\1213123\Documents\Scripts\FoundationStereo\scripts\run_files\batch_input2\seq37\step0.frame_data.json') as f: print_m(json.load(f), 'batch_input2')
