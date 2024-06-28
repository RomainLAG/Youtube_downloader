import tkinter as tk
from tkinter import filedialog, ttk
import yt_dlp as youtube_dl
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from configparser import ConfigParser
import os
import re

class YoutubeDownloaderApp:
    def __init__(self, master):
        self.master = master
        master.title("Youtube Downloader")

        self.label_url = tk.Label(master, text="URL de la playlist YouTube Music :")
        self.label_url.pack()

        self.playlist_url_entry = tk.Entry(master)
        self.playlist_url_entry.pack()

        self.label_output = tk.Label(master, text="Emplacement du dossier de réception :")
        self.label_output.pack()

        self.output_path_entry = tk.Entry(master)
        self.output_path_entry.pack()

        self.browse_button = tk.Button(master, text="Parcourir", command=self.browse_output_path)
        self.browse_button.pack()

        self.button = tk.Button(master, text="Télécharger", command=self.download_playlist)
        self.button.pack()

        self.progress_label = tk.Label(master, text="")
        self.progress_label.pack()

        # Barre de progression
        self.progress_bar = ttk.Progressbar(master, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack()

        # Initialisation de configparser et lecture de config.ini
        self.config = ConfigParser()
        config_file = os.path.join(os.getcwd(), 'config.ini')
        if not os.path.exists(config_file):
            self.progress_label.config(text="Le fichier config.ini est manquant.")
            raise FileNotFoundError("Le fichier config.ini est manquant.")
        
        self.config.read(config_file)

        # Récupération de l'emplacement de ffmpeg à partir de config.ini
        if 'settings' not in self.config or 'ffmpeg_location' not in self.config['settings']:
            self.progress_label.config(text="ffmpeg_location n'est pas configuré dans config.ini.")
            raise KeyError("ffmpeg_location n'est pas configuré dans config.ini.")
        
        self.ffmpeg_location = self.config['settings']['ffmpeg_location']
        print(f"Chemin de ffmpeg lu à partir de config.ini : {self.ffmpeg_location}")

        # Attributs pour le modèle TF-IDF
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.video_infos = None

    def browse_output_path(self):
        output_path = filedialog.askdirectory()
        self.output_path_entry.delete(0, tk.END)
        self.output_path_entry.insert(0, output_path)

    def download_playlist(self):
        playlist_url = self.playlist_url_entry.get()
        output_path = self.output_path_entry.get()
        
        if not playlist_url:
            self.progress_label.config(text="Veuillez entrer une URL de playlist")
            return
        if not output_path:
            self.progress_label.config(text="Veuillez sélectionner un emplacement de dossier de réception")
            return

        # Configuration de youtube_dl pour télécharger la playlist
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'ffmpeg_location': self.ffmpeg_location,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'extractaudio': True,
            'audioformat': 'mp3',
            'noplaylist': False,
            'ignoreerrors': True,
            'verbose': True,
            'progress_hooks': [self.progress_hook]
        }

        try:
            print(f"Options de téléchargement : {ydl_opts}")
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(playlist_url, download=True)
                # Sauvegarde des informations dans un fichier JSON
                with open('playlist_info.json', 'w', encoding='utf-8') as f:
                    json.dump(info_dict, f, ensure_ascii=False, indent=4)

            self.progress_label.config(text="Téléchargement terminé et informations sauvegardées.")

            # Traitement des données téléchargées
            self.video_infos = self.data_process('playlist_info.json')

            # Entraînement du modèle TF-IDF
            self.tfidf_matrix = self.train_model(self.video_infos)
            print(self.tfidf_matrix)  # Affichage à titre de vérification
            
        except youtube_dl.utils.DownloadError as e:
            self.progress_label.config(text=f"Erreur lors du téléchargement : {str(e)}")
        except Exception as e:
            self.progress_label.config(text=f"Erreur inattendue : {str(e)}")

    def progress_hook(self, d):
        if d['status'] == 'downloading':
            percent_str = re.sub(r'\x1b\[[0-9;]*m', '', d['_percent_str'])
            self.progress_bar['value'] = float(percent_str.strip('%'))
            self.master.update_idletasks()
        elif d['status'] == 'finished':
            self.progress_bar['value'] = 100
            self.master.update_idletasks()

    def data_process(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        video_infos = []
        for entry in data.get('entries', []):
            if entry is not None:
                video_info = {
                    'title': entry.get('title', ''),
                    'description': entry.get('description', ''),
                    'tags': entry.get('tags', [])
                }
                if video_info['title'] and video_info['description'] and video_info['tags']:
                    video_infos.append(video_info)
                else:
                    print("Warning: Entry with missing fields was found and skipped.")
            else:
                print("Warning: A 'None' entry was found in the data and skipped.")

        return video_infos

    def train_model(self, video_infos):
        titles = [info['title'] for info in video_infos]
        descriptions = [info['description'] for info in video_infos]
        tags2 = [' '.join(info['tags']) for info in video_infos]  # Join tags into a single string
        combined = [f"{title} {desc} {tags}" for title, desc, tags in zip(titles, descriptions, tags2)]

        # Vérification du contenu des documents avant la transformation
        print("Documents à transformer en TF-IDF :")
        for doc in combined:
            print(doc)
        
        final_stopwords_list = list(fr_stop) + list(en_stop)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=0.8,
            max_features=200000,
            min_df=0.2,
            stop_words=final_stopwords_list,
            use_idf=True,
            tokenizer=word_tokenize,
            ngram_range=(1, 3),
            token_pattern=None  # Explicitement définir token_pattern à None
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined)
        return tfidf_matrix

    def playlist_recommandee(self, query):
        if not self.tfidf_vectorizer or not self.tfidf_matrix:
            raise ValueError("Le modèle TF-IDF n'a pas été entraîné.")

        query_vecteur = self.tfidf_vectorizer.transform([query])
        similarite = cosine_similarity(query_vecteur, self.tfidf_matrix)
        sim_indices = np.argsort(similarite.ravel())[::-1]
        num_recommande = 10
        playlist_recommandee = []

        for i in range(num_recommande):
            video_index = sim_indices[i]
            playlist_recommandee.append(self.video_infos[video_index]['title'])

        return playlist_recommandee

    def afficher_playlist_recommandee(self, query):
        playlist = self.playlist_recommandee(query)
        fenetre_playlist = tk.Toplevel(self)
        fenetre_playlist.title("Playlist Recommandée")

        label = tk.Label(fenetre_playlist, text="Playlist Recommandée", font=("Helvetica", 16))
        label.pack(pady=10)

        for video in playlist:
            video_label = tk.Label(fenetre_playlist, text=video, wraplength=500, justify="left")
            video_label.pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = YoutubeDownloaderApp(root)
    root.mainloop()
