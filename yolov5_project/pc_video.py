import requests
import re
import json
import pprint


def get_response(html_url):
    headers={
        'user-agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
    }
    response = requests.get(url=html_url,headers=headers)
    return response

def get_video_info(html_url):
    response=get_response(html_url).text
    html_data=re.findall('<script>window.__playinfo__=(.*?)</script>',response)[0]
    json_data=json.loads(html_data)
    video_url = json_data['data']['dash']['video'][0]['baseUrl']
    audio_url = json_data['data']['dash']['audio'][0]['baseUrl']
    title = re.findall('<title data-vue-meta="true">(.*?)</title>',response)[0]
    video_info = [video_url,audio_url,title]
    return video_info

def save_video(video_info):
    video_content = get_response(video_info[0]).content
    audio_content = get_response(video_info[1]).content
    with open("./video/"+video_info[2]+".mp4",mode='wb') as f:
        f.write(video_content) 
    with open("./video/"+video_info[2]+".mp3",mode='wb') as f:
        f.write(audio_content)
    print("视频已保存") 

