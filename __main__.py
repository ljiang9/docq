"""`python -m docq` 入口。"""
try:
    from .docq import main
except ImportError:  # 直接运行 __main__.py 时的兜底
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from docq import main

if __name__ == "__main__":
    raise SystemExit(main())
