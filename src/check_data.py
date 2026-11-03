"""Aggregate inspection without printing raw dataset messages."""
import json
from src.data_utils import load_and_check

def main():
    _,report=load_and_check()
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
