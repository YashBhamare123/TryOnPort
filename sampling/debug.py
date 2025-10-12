import os, sys

print("🔍 Current working directory:", os.getcwd())
print("📁 Script directory:", os.path.dirname(__file__))
print("🧠 sys.path entries:")
for p in sys.path:
    print("   ", p)

print("📂 Files in current directory:")
print(os.listdir(os.path.dirname(__file__)))
