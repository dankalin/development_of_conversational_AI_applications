import gdown
import os

BERT_DIR = 'results'

def download_bert():
    url_bert = 'https://drive.google.com/drive/folders/155rpl9iMm_exQ9K_D7GA-PkNcbZGuy7e?usp=drive_link'
    gdown.download_folder(url_bert, output=BERT_DIR, use_cookies=False)

if __name__=='__main__':
    download_bert()