"""Evaluate HF base or LoRA adapter on wholly held-out synthetic domains.

By default uses the final-test split, which must not be used to choose model
checkpoints or training hyperparameters. Generated SQL is compared on real
fixture results, not by string exact match. Run only on CUDA GPU.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from dataset_utils import load_file
from evaluate_baseline import evaluate_case, summarize
from training.evaluate_hf import SqlGenerator
from training.train_qlora import MODEL

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'training' / 'data'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', default=MODEL)
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--limit', type=int, default=0, help='0 = complete split')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be >=0')
    rows = [json.loads(line) for line in (DATA / (args.split + '.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
    target = rows if args.limit == 0 else rows[:args.limit]
    label = 'adapter' if args.adapter else 'base'
    output = args.output or ROOT / 'reports' / f'synthetic_{args.split}_{label}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    for row in target:
        schema_id = row['schema_id']
        if schema_id not in cache:
            path = DATA / 'fixtures' / (schema_id + '.csv')
            cache[schema_id] = load_file(path.name, path.read_bytes())
    # Avoid reusing a partial result file accidentally; a full held-out test
    # needs new samples for every generation. Explicitly disallow overwrite.
    if output.exists():
        raise SystemExit(f'Results already exist: {output}. Provide --output with a new filename.')
    model_id = ('adapter:' + args.adapter) if args.adapter else ('base:' + args.base_model)
    generator = SqlGenerator(args.base_model, args.adapter)
    outcomes = []
    with output.open('x', encoding='utf-8') as fp:
        for idx, row in enumerate(target, 1):
            case = {'id': row['id'], 'dataset': row['schema_id'],
                    'difficulty': 'synthetic', 'skill': row['skill'],
                    'question': row['question'], 'gold_sql': row['sql'],
                    'ordered': 'ORDER BY' in row['sql'].upper()}
            outcome = evaluate_case(case, cache[row['schema_id']], model=model_id,
                                    url='local://hf', generate=generator.generate,
                                    repair=generator.repair)
            outcomes.append(outcome)
            fp.write(json.dumps(outcome, ensure_ascii=False) + '\n')
            fp.flush()
            if idx <= 5 or idx % 50 == 0 or idx == len(target):
                print(f'[{idx}/{len(target)}] {row["id"]}: {"PASS" if outcome["final_correct"] else "FAIL"}', flush=True)
    summ = summarize(outcomes, total=len(rows))
    summ.update({'split': args.split, 'backend': 'HF 4-bit NF4', 'model_id': model_id})
    path = output.with_name(output.stem + '_summary.json')
    path.write_text(json.dumps(summ, indent=2) + '\n', encoding='utf-8')
    print(f"Final accuracy: {summ['after_one_repair_execution_accuracy_pct']}%; {path}")


if __name__ == '__main__':
    main()
