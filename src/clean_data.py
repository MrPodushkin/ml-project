"""Canonical normalization and duplicate handling, for local inspection."""
from src.data_utils import ROOT,load_and_check

def main():
    df,report=load_and_check()
    df.to_csv(ROOT/'data/sms_clean.csv',index=False)
    print(report)

if __name__=='__main__':
    main()
