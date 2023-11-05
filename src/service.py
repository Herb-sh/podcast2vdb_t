import feed
from core import transcribe
from fastapi import FastAPI, HTTPException, Request, Response, status
from dotenv import dotenv_values
config = dotenv_values("../.env")

##uvicorn service:app --host 0.0.0.0 --port 80 --reload



app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the podcast transcription as a service API!"}


@app.get("/v1/podcast/{podcast_name}")
def feed(podcast_name: str):
    return feed.search_podcast(podcast_name)


@app.get("/v1/podcast/episodes/{podcast_id}")
async def get_episodes(podcast_id:str,  max_results:int=100, last_saved_episode:int=0):
    '''
    get all episodes of a given podcast up to limit of max_result since last_saved_episode
    '''
    return feed.get_episodes(podcast_id, max_results=max_results, since=last_saved_episode)
 

