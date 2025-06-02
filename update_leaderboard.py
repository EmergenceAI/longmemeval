import json, os, re

def main():
    datas = []
    for jsonfile in sorted(os.listdir('runs')):
        if re.match(r'^\d{8}.*\.json$', jsonfile):
            with open(f'runs/{jsonfile}', 'r') as dfile: datas.append(json.load(dfile))
    #
    
