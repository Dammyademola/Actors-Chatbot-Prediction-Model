import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

def search_images(actor_name, count=10):
    search_query = quote_plus(actor_name)
    url = f"https://www.google.com/search?q={search_query}&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    images = soup.find_all("img", limit=count)
    image_urls = [img['src'] for img in images]
    return image_urls

def download_images(image_urls, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, url in enumerate(image_urls):
        try:
            image_data = requests.get(url).content
            with open(os.path.join(output_dir, f"my_{i}.jpg"), 'wb') as f:
                f.write(image_data)
            print(f"Downloaded my_{i}.jpg")
        except Exception as e:
            print(f"Failed to download image {url}: {str(e)}")

if __name__ == "__main__":
    actor_name = input("Enter the name of the actor: ")
    num_images = int(input("Enter the number of images you want: "))

    images = search_images(actor_name, count=num_images)
    if images:
        output_directory = f"{actor_name}_images"
        download_images(images, output_directory)
        print(f"Images downloaded successfully to {output_directory}")
    else:
        print("No images found.")
