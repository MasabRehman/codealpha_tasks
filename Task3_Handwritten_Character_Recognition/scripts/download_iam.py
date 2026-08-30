import os
import requests
import tarfile
import getpass
import sys

def download_file(session, url, dest_path):
    print(f"Downloading {url}...")
    response = session.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download. Status code: {response.status_code}")
        sys.exit(1)
        
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB
    
    with open(dest_path, 'wb') as file:
        downloaded = 0
        for data in response.iter_content(block_size):
            file.write(data)
            downloaded += len(data)
            if total_size > 0:
                percent = int(50 * downloaded / total_size)
                sys.stdout.write(f"\r[{'=' * percent}{' ' * (50 - percent)}] {downloaded/1024/1024:.1f} MB")
                sys.stdout.flush()
    print("\nDownload complete.")

def extract_tar(tar_path, extract_path):
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete.")

def main():
    print("=== IAM Handwriting Database Downloader ===")
    print("Please enter your FKI portal credentials.")
    email = input("Email: ")
    password = getpass.getpass("Password: ")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    iam_dir = os.path.join(base_dir, "data", "raw", "IAM")
    os.makedirs(iam_dir, exist_ok=True)
    
    session = requests.Session()
    
    print("\nAuthenticating...")
    login_url = "https://fki.tic.heia-fr.ch/login"
    login_data = {
        '_username': email,
        '_password': password
    }
    
    # Send login POST request (FKI uses _username and _password as form fields based on standard Symfony setups, but let's send both email/password just in case)
    login_data['email'] = email
    login_data['password'] = password
    
    res = session.post(login_url, data=login_data)
    
    # Verify login by checking cookies or response
    if 'PHPSESSID' not in session.cookies.get_dict() and 'REMEMBERME' not in session.cookies.get_dict():
        print("Warning: Login might have failed. Attempting download anyway...")
        
    words_url = "https://fki.tic.heia-fr.ch/DBs/iamDB/data/words.tgz"
    ascii_url = "https://fki.tic.heia-fr.ch/DBs/iamDB/data/ascii.tgz"
    
    words_tgz = os.path.join(iam_dir, "words.tgz")
    ascii_tgz = os.path.join(iam_dir, "ascii.tgz")
    
    download_file(session, words_url, words_tgz)
    download_file(session, ascii_url, ascii_tgz)
    
    print("\nExtracting datasets into data/raw/IAM...")
    extract_tar(words_tgz, os.path.join(iam_dir, "words"))
    extract_tar(ascii_tgz, os.path.join(iam_dir, "ascii"))
    
    print("\nSuccessfully downloaded and extracted the IAM dataset!")
    print("You can now run 'python train.py' to start training on the real data.")

if __name__ == '__main__':
    main()
