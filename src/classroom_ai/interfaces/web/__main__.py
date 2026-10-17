"""Start the Streamlit adapter using the installed application package."""

import sys
from pathlib import Path


def main():
    from streamlit.web import cli

    script = str(Path(__file__).with_name("app.py"))
    sys.argv = ["streamlit", "run", script, *sys.argv[1:]]
    return cli.main()


if __name__ == "__main__":
    main()
