# ForTUNE_Teller
Song Popularity Predictor

### How to create the dataset?
<br>
1. Download the playlist by changing the URL in playlist_downloader.py .
<br>
2. Create XML file using JAudio. <br>
3. Run parser.py. Change the xml file name accordingly. <br>
4. In the csv created, change the data_type_id to corresponding youtube video id (manually). <br>
5. Run download.py
<br>
6. Repeat for every playlist. 

### How to install the dependencies?
<br>
sudo pip install --upgrade youtube_dl <br>
sudo pip install python-oauth2         <br>
sudo pip install google <br>
sudo  pip install --upgrade oauth2client <br>
sudo pip install --upgrade google-api-python-client <br>
sudo  pip install oauth2 <br>
sudo pip install google-auth <br>
sudo pip install --upgrade google-cloud <br>
sudo pip install google-auth-oauthlib  <br>
