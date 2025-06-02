# Update the leaderboard.txt file in the runs/ directory with the latest results.
import json, os, re

def main():
    datas = []
    for jsonfile in sorted(os.listdir('runs')):
        if re.match(r'^\d{8}.*\.json$', jsonfile):
            with open(f'runs/{jsonfile}', 'r') as dfile: datas.append(json.load(dfile))
    #
    dstrs = []
    for data in datas:
        dstr = ''
        dstr += f'# {data["description"]}\n'
        dstr += f'# {data["timestamp"]} {data["git"]["branch"]} {data["git"]["commit"]}\n'
        dstr += f'{data["total_time"]/500:.3f}s\n'
        dstr += f'{data["metrics"]}'
        dstrs.append(dstr)

    # Hack.  Read in leaderboard.txt
    with open('leaderboard.txt', 'r') as lfile: leaderboard = lfile.read()
    # Split on blank lines (might be more than one)
    leaderboard = re.split(r'\n\s*\n', leaderboard.strip())
    leaderboard += dstrs
    # Sort by accuracy
    leaderboard = sorted(leaderboard, key=lambda x: float(re.search(r'Accuracy:\s*([\d.]+)', x).group(1)), reverse=True)

    # Write a new leaderboard.txt in runs/ .  Delete the old one if it exists.
    with open('runs/leaderboard.txt', 'w') as lfile:
        lfile.write('\n\n'.join(leaderboard))
   
if __name__ == '__main__': main()
