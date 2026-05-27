"""
Helper script to get Google Drive file IDs
Run this after uploading files to Google Drive to get the file IDs
"""
import re

def get_file_id_from_url(url):
    """
    Extract file ID from Google Drive URL
    
    Examples:
    - https://drive.google.com/file/d/1aBcD_EfgHiJkLmNoPqRsTuvWxYz/view?usp=sharing
    - https://drive.google.com/open?id=1aBcD_EfgHiJkLmNoPqRsTuvWxYz
    - https://drive.google.com/uc?id=1aBcD_EfgHiJkLmNoPqRsTuvWxYz
    """
    # Pattern 1: /file/d/FILE_ID/view
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    # Pattern 2: ?id=FILE_ID
    match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    return None

# Instructions
print("=" * 60)
print("Google Drive File ID Extractor")
print("=" * 60)
print("\n📋 Instructions:")
print("1. Upload your vector store files to Google Drive:")
print("   - chunks.json")
print("   - config.json")
print("   - faiss_index.bin")
print("\n2. Right-click each file → 'Get link' → 'Anyone with the link'")
print("\n3. Paste the shareable links below:")
print("\n" + "-" * 60)

# Get file IDs from user
chunks_url = input("\n📄 chunks.json URL: ").strip()
config_url = input("📄 config.json URL: ").strip()
faiss_url = input("📄 faiss_index.bin URL: ").strip()

# Extract file IDs
chunks_id = get_file_id_from_url(chunks_url) if chunks_url else None
config_id = get_file_id_from_url(config_url) if config_url else None
faiss_id = get_file_id_from_url(faiss_url) if faiss_url else None

print("\n" + "=" * 60)
print("File IDs extracted:")
print("=" * 60)

if chunks_id:
    print(f"\nGDRIVE_CHUNKS_JSON_ID = \"{chunks_id}\"")
else:
    print("\n⚠️  Could not extract chunks.json file ID")

if config_id:
    print(f"GDRIVE_CONFIG_JSON_ID = \"{config_id}\"")
else:
    print("⚠️  Could not extract config.json file ID")

if faiss_id:
    print(f"GDRIVE_FAISS_INDEX_ID = \"{faiss_id}\"")
else:
    print("⚠️  Could not extract faiss_index.bin file ID")

print("\n" + "=" * 60)
print("📝 Next steps:")
print("=" * 60)
print("1. Copy the file IDs above")
print("2. Update initialize.py with these IDs")
print("3. Make sure the files are set to 'Anyone with the link can view'")
print("4. Push to GitHub and deploy!")

