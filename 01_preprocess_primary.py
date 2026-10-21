"""Deterministic cleaning, feature engineering and stratified 70/15/15 split."""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline_common import RAW_FEATURES, SEED, derive

def main():
    Path('outputs').mkdir(exist_ok=True)
    df = pd.read_csv('data/cardio_train.csv', sep=';')
    audit = {'raw_records': len(df), 'raw_class_counts': df.cardio.value_counts().to_dict()}
    df = df.drop(columns=['id'])
    # Identify duplicates on original precision, before age rounding or derivation.
    # Drop all repeated predictor profiles: prevents identical profiles across splits,
    # including profiles with contradictory labels, without choosing a preferred label.
    repeated = df.duplicated(subset=RAW_FEATURES, keep=False)
    audit['repeated_profile_rows_removed'] = int(repeated.sum())
    audit['repeated_profiles_with_conflicting_labels'] = int((df.groupby(RAW_FEATURES).cardio.nunique()>1).sum())
    df = df.loc[~repeated].copy()
    valid = (df.ap_hi.gt(60)&df.ap_hi.lt(240)&df.ap_lo.gt(40)&df.ap_lo.lt(200)&df.ap_hi.ge(df.ap_lo)&df.height.gt(100)&df.height.lt(220)&df.weight.gt(30)&df.weight.lt(200))
    audit['implausible_rows_removed_after_deduplication'] = int((~valid).sum())
    df = df.loc[valid].copy()
    df['age'] = df.age / 365.25
    df = derive(df)
    train_val, test = train_test_split(df,test_size=.15,stratify=df.cardio,random_state=SEED)
    train, val = train_test_split(train_val,test_size=.15/.85,stratify=train_val.cardio,random_state=SEED)
    splits = {'train':train,'val':val,'test':test}
    profiles = {k:set(map(tuple,v[RAW_FEATURES].to_numpy())) for k,v in splits.items()}
    assert not (profiles['train'] & profiles['val'] or profiles['train'] & profiles['test'] or profiles['val'] & profiles['test'])
    audit.update(clean_records=len(df), clean_class_counts=df.cardio.value_counts().to_dict(), random_state=SEED, balancing='None; natural class distribution retained', split_strategy='Random row-wise stratified 70/15/15; subject identifiers unavailable', cross_split_duplicate_profiles=0)
    audit['splits'] = {}
    for name, part in splits.items():
        part.to_csv(f'data/primary_{name}.csv',index=False)
        audit['splits'][name] = {'n':len(part),'class_counts':part.cardio.value_counts().to_dict()}
    Path('outputs/preprocessing_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__ == '__main__': main()
