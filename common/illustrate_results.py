import glob
import json
import os
import time

from prettytable import PrettyTable


def monitor():
    total_table = []
    for dataset in ['val']:
        table = PrettyTable(['exp_name', 'exp_id', 'accuracy'],
                            sortby='accuracy',
                            header_style='upper',
                            valign='m',
                            title='{} Result'.format(dataset),
                            reversesort=True)

        valid_dirs = [
            'experiment_yolox_single_scale',
        ]
        exp_dirs = glob.glob('./experiments/*/exp_*')
        exp_dirs = [i for i in exp_dirs if i.split('/')[-2] in valid_dirs]

        for exp_dir in exp_dirs:
            params_json_path = os.path.join(exp_dir, 'params.json')
            results_json_path = os.path.join(exp_dir, 'val_metrics_best.json')
            logs_txt_path = os.path.join(exp_dir, 'log.txt')
            if not os.path.exists(params_json_path) or not os.path.exists(
                    results_json_path):
                continue

            params = json.load(open(params_json_path, 'r'))
            results = json.load(open(results_json_path, 'r'))
            # exp info
            exp_name = exp_dir.split('/')[-2]
            exp_id = exp_dir.split('_')[-1]
            # model = params['model']
            # results
            accuracy = '{:>8.4f}'.format(results['accuracy'])
            cur_row = [exp_name, exp_id, accuracy]
            table.add_row(cur_row)
        print(table)
        total_table.append(str(table))


def run(interval):
    while True:
        monitor()
        time.sleep(interval)


if __name__ == '__main__':
    interval = 10 * 60
    run(interval)
