import pandas as pd
import requests
from langdetect import detect, LangDetectException
import requests
import time
import random
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


# movies = [
#     "Staying Alive",
#     "Bolero",
#     "Problem Child",
#     "Return to the Blue Lagoon",
#     "Simon Sez",
#     "3 Strikes",
#     "Killing Me Softly",
#     "Merci Docteur Rey",
#     "Pinocchio",
#     "Derailed",
#     "Superbabies: Baby Geniuses 2",
#     "Constellation",
#     "Redline",
#     "Scar",
#     "One Missed Call",
#     "Stolen",
#     "Transylmania",
#     "Beneath the Darkness",
#     "Dark Tide",
#     "A Thousand Words",
#     "The Ridiculous 6",
#     "Cabin Fever",
#     "Dark Crimes",
#     "The Disappointments Room",
#     "Max Steel",
#     "Stratton",
#     "Gotti",
#     "London Fields",
#     "The Queen's Corgi",
#     "The Last Days of American Crime",
#     "Hard Kill",
#     "Armor",
#     "Alarum",
#     "Loqueesha",
#     "The Emoji Movie",
#     "Fred: The Movie",
#     "The Garbage Pail Kids Movie",
#     "Best Night Ever",
#     "Assassin's Bullet",
#     "The Faith of Anna Waters",
#     "Sundown",
#     "All Roads Lead to Rome",
#     "A Haunting at Silver Falls",
#     "The Human Centipede 3",
#     "The Hungover Games",
#     "InAPPropriate Comedy",
#     "Kirk Cameron's Saving Christmas",
#     "The Legend of Hercules",
#     "Meet the Spartans",
#     "Movie 43",
#     "The Open House",
#     "Paranormal Activity: The Ghost Dimension",
#     "The Perfect Man",
#     "The Prince",
#     "The Roommate",
#     "Shark Night",
#     "Slender Man",
#     "Space Chimps 2: Zartog Strikes Back",
#     "Spy Kids 4: All the Time in the World",
#     "Stealth",
#     "Street Fighter: The Legend of Chun-Li",
#     "Supernova",
#     "Surfer, Dude",
#     "Taking 5",
#     "The 41-Year-Old Virgin Who Knocked Up Sarah Marshall and Felt Superbad About It",
#     "The Apparition",
#     "The Black Dahlia",
#     "The Book of Henry",
#     "The Boy Next Door",
#     "The Bye Bye Man",
#     "The Circle",
#     "The Covenant",
#     "The Darkness",
#     "The Fanatic",
#     "The Gallows",
#     "The Happening",
#     "The Haunting of Molly Hartley",
#     "The Hottie and the Nottie",
#     "The In Crowd",
#     "The Intruder",
#     "The Kitchen",
#     "The Last Airbender",
#     "The Messengers",
#     "The Mummy: Tomb of the Dragon Emperor",
#     "The Number 23",
#     "The Oogieloves in the Big Balloon Adventure",
#     "The Perfect Guy",
#     "The Pink Panther 2",
#     "The Reaping",
#     "The Roommate",
#     "The Secret Life of Pets 2",
#     "The Snowman",
#     "The Stepfather",
#     "The Tuxedo",
#     "The Ugly Truth",
#     "The Unborn",
#     "The Watch",
#     "The Woman in the Window",
#     "The Wretched",
#     "The Zookeeper's Wife"
#     "Call Me By Your Name",
#     "The Godfather",
#     "Pulp Fiction",
#     "The Dark Knight",
#     "Schindlers List",
#     "Inception",
#     "Fight Club",
#     "Forrest Gump",
#     "Matrix",
#     "The Lord of the Rings The Fellowship of the Ring",
#     "Interstellar 2014",
#     "Hocus Pocus",
#     "Avengers Endgame",
#     "Parasite",
#     "Whiplash",
#     "Gladiator",
#     "Silence of the Lambs",
#     "Titanic",
#     "Goodfellas",
#     "Mad Max Fury Road",
#     "The Wolf of Wall Street 2013",
#     "La La Land",
#     "12 Years a Slave",
#     "Birds Of Prey 2020",
#     "The Revenant 2015",
#     "Joker",
#     "Inglourious Basterds",
#     "The Grand Budapest Hotel",
#     "Her",
#     "Birdman 2014",
#     "The Big Short",
#     "A Star is Born 2018",
#     "Bohemian Rhapsody",
#     "Black Panther",
#     "Guardians of the Galaxy",
#     "Logan",
#     "Get Out",
#     "Moonlight",
#     "The Shape of Water 2017",
#     "Lady Bird",
#     "Arrival",
#     "No Country for Old Men",
#     "It Ends With Us",
#     "Gone Girl",
#     "A Quiet Place",
#     "Shutter Island",
#     "The Prestige",
#     "The Departed",
#     "There Will Be Blood",
#     "Eternal Sunshine of the Spotless Mind",
#     "The Truman Show",
#     "Casino Royale",
#     "The Bourne Ultimatum",
#     "Skyfall",
#     "The Kings Speech",
#     "Slumdog Millionaire",
#     "The Pianist",
#     "The Green Mile",
#     "The Sixth Sense",
#     "American Beauty",
#     "Requiem for a Dream",
#     "The Imitation Game",
#     "The Theory of Everything",
#     "Argo",
#     "The Hurt Locker",
#     "The Pursuit of Happyness",
#     "Catch Me If You Can",
#     "Curious Case of Benjamin Button",
#     "V for Vendetta",
#     "Bodies Bodies Bodies",
#     "Star Wars The Force Awakens",
#     "Jurassic Park",
#     "Indiana Jones and the Last Crusade",
#     "The Avengers",
#     "Iron Man",
#     "The Hunger Games",
#     "Divergent",
#     "The Maze Runner",
#     "Terrifier",
#     "The Amazing Spider-Man",
#     "Man of Steel",
#     "Wonder Woman",
#     "Doctor Strange",
#     "Thor Ragnarok 2017",
#     "AntMan",
#     "Captain America Civil War",
#     "Star Trek 11",
#     "Pacific Rim",
#     "Transformers",
#     "Live Die Repeat Edge Of Tomorrow",
#     "Ready Player One",
#     "Mission Impossible Fallout",
#     "The Lego Movie",
#     "Frozen 2013",
#     "Zootopia",
#     "Moana",
#     "Coco",
#     "Inside Out",
#     "Toy Story 3",
#     "Up",
#     "Finding Nemo",
#     "Monsters Inc",
#     "Ratatouille",
#     "The Incredibles"
# ]

HEADERS = {
    'User-Agent': 'Mozilla/5.0'
}

SAVE_FILE = "cached_reviews.json"

def extract_reviews_and_scores(soup):
    review_elements = soup.find_all('p', class_='audience-reviews__review js-review-text')
    score_elements = soup.find_all('rating-stars-group')
    reviews_scores = []
    for i, review_elem in enumerate(review_elements):
        text = review_elem.get_text(strip=True)
        score_tag = score_elements[i] if i < len(score_elements) else None
        score = score_tag.get('score') if score_tag else None
        reviews_scores.append((text, score))
    return reviews_scores

def fetch_page(session, url, params=None):
    for i in range(3):
        try:
            time.sleep(random.uniform(0.1, 0.3))  # tiny polite delay
            response = session.get(url, headers=HEADERS, params=params, timeout=10)
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
        except RequestException:
            time.sleep(2 ** i)
    return None


def process_movie(movie_slug, target_each, cache, max_pages=10):
    pos_reviews, neg_reviews = [], []
    session = requests.Session()
    page = 1
    base_url = f"https://www.rottentomatoes.com/m/{movie_slug}/reviews"

    while (len(pos_reviews) < target_each or len(neg_reviews) < target_each) and page <= max_pages:
        soup = fetch_page(session, base_url, params={'type': 'user', 'page': page})
        if not soup:
            print(f"{movie_slug}: Failed to fetch page {page}, skipping to next movie.")
            break
        reviews_scores = extract_reviews_and_scores(soup)
        if not reviews_scores:
            print(f"{movie_slug}: No reviews found on page {page}, stopping.")
            break

        for text, score in reviews_scores:
            if len(text) < 5:
                continue
            try:
                if detect(text) != 'en':
                    continue
            except LangDetectException:
                continue
            try:
                score = float(score)
            except (ValueError, TypeError):
                continue
            if score == 3 or score is None:
                continue
            if score > 3 and len(pos_reviews) < target_each:
                pos_reviews.append((text, score))
            elif score < 3 and len(neg_reviews) < target_each:
                neg_reviews.append((text, score))
        page += 1

    print(f"{movie_slug}: Got {len(pos_reviews)} positive and {len(neg_reviews)} negative reviews.")

    # Save everything, even if incomplete
    cache[movie_slug] = {
        "pos": pos_reviews,
        "neg": neg_reviews
    }

    return pos_reviews, neg_reviews


def get_reviews_parallel(movies, amount):
    target_each = amount // 2
    reviews_needed = target_each // len(movies)

    # Load cache
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    pos_total, neg_total = [], []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for name in movies:
            movie_slug = name.replace(' ', '_').lower()
            if movie_slug in cache:
                print(f"Using cached reviews for {movie_slug}")
                pos = cache[movie_slug]["pos"]
                neg = cache[movie_slug]["neg"]
                pos_total.extend(pos[:reviews_needed])
                neg_total.extend(neg[:reviews_needed])
            else:
                futures[executor.submit(process_movie, movie_slug, reviews_needed, cache)] = movie_slug

        for future in as_completed(futures):
            try:
                pos, neg = future.result()
                pos_total.extend(pos[:reviews_needed])
                neg_total.extend(neg[:reviews_needed])
            except Exception as e:
                print(f"Error with movie {futures[future]}: {e}")

    # Save cache
    with open(SAVE_FILE, 'w') as f:
        json.dump(cache, f)

    # If imbalance remains, reprocess cached data to try to balance
    if len(pos_total) < target_each or len(neg_total) < target_each:
        print("⚠️  Imbalance detected. Reprocessing cached reviews to balance...")

        unused_pos = []
        unused_neg = []
        for data in cache.values():
            unused_pos.extend(data["pos"])
            unused_neg.extend(data["neg"])

        random.shuffle(unused_pos)
        random.shuffle(unused_neg)

        while len(pos_total) < target_each and unused_pos:
            pos_total.append(unused_pos.pop())

        while len(neg_total) < target_each and unused_neg:
            neg_total.append(unused_neg.pop())

        print(f"📊 Final counts: {len(pos_total)} positive, {len(neg_total)} negative")

    # Limit to desired amount
    pos_total = pos_total[:target_each]
    neg_total = neg_total[:target_each]
    all_reviews = [r[0] for r in pos_total + neg_total]
    all_ratings = [r[1] for r in pos_total + neg_total]
    return all_reviews, all_ratings



if __name__ == "__main__":
    with open("movies.json") as f:
        movies = json.load(f)

    reviews, ratings = get_reviews_parallel(movies, amount=20000)

    df = pd.DataFrame({
    'review': reviews,
    'rating': ratings
    })

    df.to_csv('../../data/movie_reviews.csv', index=False)