"""Downloads open-licence (OFL) fonts from google/fonts."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl"
FONTS = {
    "Anton-Regular.ttf": f"{BASE}/anton/Anton-Regular.ttf",
    "Poppins-SemiBold.ttf": f"{BASE}/poppins/Poppins-SemiBold.ttf",
    "Poppins-Bold.ttf": f"{BASE}/poppins/Poppins-Bold.ttf",
    "Poppins-LightItalic.ttf": f"{BASE}/poppins/Poppins-LightItalic.ttf",
    "BebasNeue-Regular.ttf": f"{BASE}/bebasneue/BebasNeue-Regular.ttf",
    "PlayfairDisplay.ttf": f"{BASE}/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "Caveat-Bold.ttf": f"{BASE}/caveat/Caveat%5Bwght%5D.ttf",
}

here = Path(__file__).parent
for filename, url in FONTS.items():
    dest = here / filename
    if not dest.exists():
        try:
            urllib.request.urlretrieve(url, dest)
            print("downloaded", filename)
        except Exception as e:
            print("failed", filename, e)
    else:
        print("already exists", filename)
