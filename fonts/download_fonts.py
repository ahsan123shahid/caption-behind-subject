"""Downloads open-licence (OFL) fonts from google/fonts."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl"
FILES = {"Anton-Regular.ttf": "anton", "Poppins-SemiBold.ttf": "poppins",
         "Poppins-Bold.ttf": "poppins", "Poppins-LightItalic.ttf": "poppins"}
here = Path(__file__).parent
for f, d in FILES.items():
    urllib.request.urlretrieve(f"{BASE}/{d}/{f}", here / f)
    print("ok", f)
