import easygui,nltk,urllib,pyaudio,wave,sys,urllib2,commands,os
from nltk import word_tokenize,pos_tag,ne_chunk
import xml.etree.ElementTree as xml
address='Not Available'
name='Not Available'
rating='Not Available'
hour='Not Available'
google_api='AIzaSyDYXoQuVQVaMMtqkJTppwvMh92-jWzKnEA'
location=[('i want some dinner','location'),('find me some food','location'),('Where','location'),('Where is a restaurant','location'),('Get directions to the nearest restaurant','location'),
          ('I want a soccer ball','location'),('I want some food','location'),
          ('Find Walmart','location'),('Find coffee near me','location'),
          ('Where is Starbucks','location'),('Find some burger joints in Melbourne','location'),
          ('Find a gas station near me','location'),('Good Indian restaurants around here','location'),
          ('Find me pizza in Dallas','location'),('Show me nearby Italian restaurants','location'),
          ('Can I get some food around here','location'),('What Chinese restaurants are near me','location'),
          ('Where is a grocery store','location'),('Get some computer stores in Melbourne','location'),
          ('Show my current location','location'),('Show me Statue of Liberty','location'),
          ('What is the best restaurant near me','location'),('Navigate to Melbourne, Florida','location'),
          ('Find my way to home','location'),('Find a locksmith','location'),('Find a clothing store','location'),
          ('Get a clothing store','location'),('Show some stores','location'),('Take me to the stadium','location'),
          ('Show me some hotels','location'),('Can you take me to a book store','location'),
          ('Can you show me the nearest Mcdonalds','location'),('where is the','location'),
          ('Get restaurants in this area','location'),('I need some new glasses','location'),('Where is the beach','location'),
          ('How far is a movie theater','location'),('Where is the nearest restaurant','location'),
          ('Can you find me a phone shop','location'),('find me some dinner','location'),
          ('find me some lunch','location'),('find me some breakfast','location')]
weather=[("weather","weather"),("What's the weather today","weather"),("Forecast for Melbourne,FL","weather"),
         ("Current condition","weather"),("Get a forecast","weather"),("Temperature for Melbourne","weather"),
         ("Current temperature for San Diego","weather"),("Condition","weather"),
         ("Weather information for Melbourne","weather"),("Is it going to rain today","weather"),
         ("Do I need an umbrella today","weather"),("Is it going to snow today","weather"),
         ("How is the weather in New York, NY","weather"),("Will it be hot in San Fransisco today","weather"),
         ("Do I need a coat today","weather"),("Will it rain","weather"),("What is the high today","weather"),
         ("What is the low today","weather"),("Will it be cold tomorrow","weather"),
         ("What is the temperature right now","weather"),("What does it feel like","weather"),
         ("Weather","weather"),("get conditions for Miami, Florida","weather"),
         ("Will it snow in New York","weather"),("What is the forecast in Chicago, Illinois","weather"),
         ("Forecast","weather"),("How hot will it be in Tampa today","weather"),
         ("What's the high for Anchorage on Thursday","weather"),
         ("How windy is it out there","weather"),("When is the sunrise in Paris","weather"),
         ("weather for tampa bay","weather"),("What's the forecast for today","weather"),
         ("Do I need a sweater","weather"),("Will I need gloves","weather"),('wind','weather'),
         ('Is there going to be a storm today','weather'),('will i need a jacket','weather'),
         ('can i wear a jacket','weather'),('can i wear a shirt','weather'),('can i wear shorts','weather'),
         ('what is the weather like','weather'),('get me the temperature','weather'),
         ('what will the weather be like','weather'),('should i wear boots','weather'),('will i need an umbrella','weather'),
         ('do i need an umbrella','weather'),('how is the wind today','weather'),
         ('whats the weather this week','weather'),('when is sunset','weather'),
         ('when is sunrise','weather'),('what time will sunrise be at','weather'),
         ('what time will sunset be at','weather'),('get me the sunset','weather'),
         ('sunset','weather'),('is the weather nice today','weather')]
general=[("weather","general"),("hows the weather for today","general"),
                 ("weather information for","general"),("how is the weather","general"),
                 ("weather for","general"),('whats the weather like today','general'),
                 ('weather','general'),('hows the weather','general'),('whats the weather today','general'),
         ('hows the weather today','general'),('get me the weather','general'),('weather','general')]
wind=[('how windy is it out there','wind'),('whats the wind right now','wind'),
      ('can you tell me the wind today','wind'),('wind','wind'),('how does it feel outside','wind'),
      ('how is the wind','wind')]
temp=[("temperature","temp"),("current temperature","temp"),
      ("high","temp"),("low","temp"),('hows the temperature','temp'),
      ('whats the temperature today','temp'),('whats the temperature','temp'),
      ('whats the low today','temp'),('whats the high today','temp'),
      ('get me the temperature','temp'),('how is it outside','temp'),('temperature','temp')]
forecast=[('whats the forecast for this week','forecast'),
          ('how the weather this week','forecast'),('get me the forecast','forecast'),
          ('monday','forecast'),('tuesday','forecast'),('wednesday','forecast'),
          ('thursday','forecast'),('friday','forecast'),('saturday','forecast'),('sunday','forecast'),
          ('weekend','forecast'),('tomorrow','forecast'),('forecast','forecast'),
          ('how is it this week','forecast'),('get a forecast','forecast')]
question=[("do I need a coat","question"),("will I need a jacket today","question"),
           ("can I dress up in shorts today","question"),('will it rain','question'),
           ("is it a good day to go to the beach","question"),("will I need gloves","question"),
           ("do i need suntan lotion today","question"),('can i dress up in a shirt','question'),
          ("will i need a hoodie","question"),('is there going to be a storm','question'),
          ('will i be able to wear a sweater','question'),('can i wear shorts','question'),
          ('can i wear a jacket','question'),('do i need a coat','question'),
          ('is it going to be sunny','question')]
queries=[ ]
for (words, sentiment) in location+weather:
    words_filtered=[e.lower() for e in words.split() if len(e)>=3]
    queries.append((words_filtered,sentiment))
def get_features(wordlist):
    wordlist=nltk.FreqDist(wordlist)
    word_features=wordlist.keys()
    return word_features
def get_words(queries):
    all_words=[]
    for (words, sentiment) in queries:
        all_words.extend(words)
    return all_words
word_features=get_features(get_words(queries))
def extract_features(thing):
    document_words=set(thing)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in document_words)
    return features
training_set=nltk.classify.apply_features(extract_features, queries)
classifier=nltk.NaiveBayesClassifier.train(training_set)
q=[ ]
for (words,sentiment) in general:
    words_filtered=[e.lower() for e in words.split() if len(e)>=3]
    q.append((words_filtered,sentiment))
for (words,sentiment) in wind:
    words_filtered=[e.lower() for e in words.split() if len(e)>=3]
    q.append((words_filtered,sentiment))
for (words,sentiment) in temp:
    words_filtered=[e.lower() for e in words.split() if len(e)>=3]
    q.append((words_filtered,sentiment))
for (words,sentiment) in forecast:
    words_filtered=[e.lower() for e in words.split() if len(e)>=3]
    q.append((words_filtered,sentiment))
for (words,sentiment) in question:
    words_filtered=[e.lower() for e in words.split() if len(e)>=3]
    q.append((words_filtered,sentiment))

word_features=get_features(get_words(q))
weather_set=nltk.classify.apply_features(extract_features, q)
weather_classifier=nltk.NaiveBayesClassifier.train(weather_set)
while 1:
    i=easygui.enterbox('Enter a query',default='speech')
    g=i.lower()
    if 'speech' in g:
        import pyaudio
        import wave
        import sys
        import urllib2
        import commands
        url="https://www.google.com/speech-api/v1/recognize?client=chromium&lang=en-us"
        header = {'Content-Type' : 'audio/x-flac; rate=44100'}
        chunk = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = 4
        WAVE_OUTPUT_FILENAME = "demo.wav" 

        p = pyaudio.PyAudio()

        stream = p.open(format = FORMAT,
                        channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        frames_per_buffer = chunk)
        all = []
        for i in range(0, RATE / chunk * RECORD_SECONDS):
            data = stream.read(chunk)
            all.append(data)

        stream.close()
        p.terminate()

        # write data to WAVE file
        data = ''.join(all)
        x=1
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()
        commands.getstatusoutput('/Users/arifadil/Downloads/sox-14.4.1/sox demo.wav output.flac')
        f=open('output.flac','rb').read()
        req=urllib2.Request(url,f,header)
        data=urllib2.urlopen(req)
        x=data.read()
        i=x.split('utterance":"')[1].split(',')[0].strip('"')
        easygui.msgbox(i)
    if i=='quit':
        break
    i=i.replace('get','')
    i=i.replace('?','')
    typeof=classifier.classify(extract_features(i.split()))
    list_1=[ ]
    token=word_tokenize(i)
    tagged=pos_tag(token)
    thing=ne_chunk(tagged)
    x=0
    y=0
    result=len(thing)-1
    l=[ ]
    z=0
    for t in thing:
        try:
            if t.node=='GPE':
                z=z+1
        except:
            continue
    if z==0:
        for t in tagged:
            if t[1]=='NN':
                list_1.append(t[0].capitalize())
            else:
                list_1.append(t[0])
        thing=nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(" ".join(list_1))))
    h=0
    for t in thing:
        try:
            if t.node=='GPE':
                z=z+1
        except:
            continue
    b=0
    if z==0:
        while 1:
            if b=='go':
                break
            try:
                appid='57WKXU-8YJQVTL8AP'
                url='http://api.wolframalpha.com/v2/query?appid='+appid+'&input=where%20am%20i&format=plaintext'
                f=urllib.urlopen(url)
                s=f.read()
                root=xml.fromstring(s)
                location=root[2][0][0].text
                location=location.split('\n')[1]
                location=location.strip('location | ')
                b='go'
            except:
                pass
    else:
        while x<= result:
            try:
                if 'GPE' in thing[x].node:
                    for t in thing[x]:
                        l.insert(4,t[0])
                    y=y+1
                    x=x+1
                    location=" ".join(l)
            except:
                x=x+1
    easygui.msgbox(location)
    if 'location' in typeof:
        easygui.msgbox('location')
        i=i.lower()
        if z>0:
            for a in location.lower().split():
                if a in i:
                    i=i.replace(a,'')
        thing=ne_chunk(pos_tag(word_tokenize(i)))
        list_1=[ ]
        for t in thing:
            try:
                if 'NN' in t[1]:
                    list_1.append(t[0])
                if 'JJ' in t[1]:
                    list_1.append(t[0])
            except:
                pass
        list_d=list_1
        if z==0:
            for t in location.split():
                list_1.append(t)
        query="+".join(list_1)
        map_url="https://maps.googleapis.com/maps/api/place/textsearch/xml?query="+query+"&sensor=false&key="+google_api
        f=urllib.urlopen(map_url)
        s=f.read()
        root=xml.fromstring(s)
        origin=location.replace(' ','+')
        x=1
        if 'OK' in root[0].text :
            easygui.msgbox('OK')
            while 1:
                try:
                    name=root[x][0].text
                    for t in root[x]:
                        if t.tag=='formatted_address':
                            address=t.text
                        if t.tag=='rating':
                            rating=t.text
                        if t.tag=='opening_hours':
                            hour=t[0].text
                    if 'true' in hour:
                        hour='Open'
                    else:
                        hour='Closed'
                    if name is not " ":
                        if 'directions' in i:
                            easygui.msgbox('directions')
                            b='true'
                            list_d=' '.join(list_d)
                            list_d=list_d.replace('directions','')
                            list_d=list_d.replace('get','')
                            if len(list_d.split())==0:
                                b='false'
                            if b=='true':
                                for v in list_d:
                                    try:
                                        if 'NN' in v[1]:
                                            b='true'
                                            break
                                    except:
                                        continue
                            if b=='false':
                                easygui.msgbox('Getting directions to '+location)
                                appid='57WKXU-8YJQVTL8AP'
                                url='http://api.wolframalpha.com/v2/query?appid='+appid+'&input=where%20am%20i&format=plaintext'
                                f=urllib.urlopen(url)
                                s=f.read()
                                root5=xml.fromstring(s)
                                location=root5[2][0][0].text
                                location=location.split('\n')[1]
                                location=location.strip('location | ')
                                location=str(location).replace(' ','+')
                                if 'from' and 'to' in i:
                                    r=i.split('from')
                                    r=' '.join(r).split('to')
                                    location=r[1]
                                    origin=r[2]
                                origin=str(origin).replace(' ','+')
                                location=str(location).replace(' ','+')
                                d_url='http://maps.googleapis.com/maps/api/directions/xml?origin='+location+'&destination='+origin+'&sensor=false&avoid=tolls'
                                f=urllib.urlopen(d_url)
                                s=f.read()
                                root3=xml.fromstring(s)
                                y=1
                                while 1:
                                    try:
                                        directions=root3[1][1][y][5].text
                                        directions=directions.replace('<b>','')
                                        directions=directions.replace('</b>','')
                                        directions=directions.replace('<div style="font-size:0.9em">',' ')
                                        directions=directions.replace('<\div>','')
                                        easygui.msgbox(directions)
                                        y=y+1
                                    except:
                                        break
                            if b=='true':
                                appid='57WKXU-8YJQVTL8AP'
                                url='http://api.wolframalpha.com/v2/query?appid='+appid+'&input=where%20am%20i&format=plaintext'
                                f=urllib.urlopen(url)
                                s=f.read()
                                root5=xml.fromstring(s)
                                location=root5[2][0][0].text
                                location=location.split('\n')[1]
                                location=location.strip('location | ')
                                location=str(location).replace(' ','+')
                                origin=origin.replace(',','')
                                origin=origin.replace(' ','+')
                                d_url='http://maps.googleapis.com/maps/api/directions/xml?origin='+origin+'&destination='+address+'&sensor=false&avoid=tolls'
                                f=urllib.urlopen(d_url)
                                s=f.read()
                                root3=xml.fromstring(s)
                                y=1
                                easygui.msgbox('Getting directions to '+name)
                                while 1:
                                    try:
                                        directions=root3[1][1][y][5].text
                                        directions=directions.replace('<b>','')
                                        directions=directions.replace('</b>','')
                                        directions=directions.replace('<div style="font-size:0.9em">',' ')
                                        directions=directions.replace('<\div>','')
                                        easygui.msgbox(directions)
                                        y=y+1
                                    except:
                                        break
                        
                        if address is not 'Not Available' and 'directions' not in i:
                            matrix_url='http://maps.googleapis.com/maps/api/distancematrix/xml?origins='+origin+'&destinations='+address+'&avoid=tolls&sensor=false'
                            f=urllib.urlopen(matrix_url)
                            s=f.read()
                            root2=xml.fromstring(s)
                            distance=root2[3][0][2][1].text
                            time=root2[3][0][1][1].text
                        choice=easygui.buttonbox(msg="Name: "+name+'\nDistance:'+distance+'\nTime:'+time+'\nAddress: '+address+'\nRating: '+rating+'\nOpening Times: '+hour,choices=['Next','Back','Quit','Directions'])
                        if choice=='Directions':
                            d_url='http://maps.googleapis.com/maps/api/directions/xml?origin='+origin+'&destination='+address+'&sensor=false&avoid=tolls'
                            f=urllib.urlopen(d_url)
                            s=f.read()
                            root3=xml.fromstring(s)
                            y=1
                            while 1:
                                try:
                                    directions=root3[1][1][y][5].text
                                    directions=directions.replace('<b>','')
                                    directions=directions.replace('</b>','')
                                    directions=directions.replace('<div style="font-size:0.9em">',' ')
                                    directions=directions.replace('<\div>','')
                                    easygui.msgbox(directions)
                                    y=y+1
                                except:
                                    break
                        if choice=='Next':
                            x=x+1
                        if choice=='Quit':
                            break
                        if choice=="Back":
                            x=x-1
                except:
                    break
    if 'weather' in typeof:
        easygui.msgbox('weather')
        weather_type=weather_classifier.classify(extract_features(i.split()))
        easygui.msgbox(weather_type)
        l=location.replace(' ','+')
        url='http://api.wolframalpha.com/v2/query?appid=57WKXU-8YJQVTL8AP&input='+l+'+zipcode&format=plaintext'
        f=urllib.urlopen(url)
        s=f.read()
        root=xml.fromstring(s)
        zipcode=root[1][0][0].text
        n=0
        for h in zipcode.split():
            try:
                int(h)
                if n==1:
                    code=h
                    break
                n=n+1
            except:
                pass
        url='http://xml.weather.yahoo.com/forecastrss/'+code+'_f.xml'
        f=urllib.urlopen(url)
        s=f.read()
        week={'1':'day1_condition','2':'day2_condition','3':'day3_condition','4':'day4_condition'}
        root=xml.fromstring(s)
        conditions=root[0][12][0].text
        hl=root[0][12][7].attrib
        high=hl['high']
        low=hl['low']
        day=hl['day']
        day_condition=hl['text']
        current=root[0][12][5].attrib
        current_condition=current['text']
        t=current['temp']
        wind=root[0][8].attrib
        sunset=root[0][10].attrib['sunset']
        sunrise=root[0][10].attrib['sunrise']
        speed=wind['speed']
        chill=wind['chill']
        day1=root[0][12][8].attrib
        day1_high=day1['high']
        day1_low=day1['low']
        day1_date=day1['date']
        day1_condition=day1['text']
        day1_day=day1['day']
        day2=root[0][12][9].attrib
        day2_high=day2['high']
        day2_low=day2['low']
        day2_date=day2['date']
        day2_day=day2['day']
        day2_condition=day2['text']
        day3=root[0][12][10].attrib
        day3_high=day3['high']
        day3_low=day3['low']
        day3_date=day3['date']
        day3_condition=day3['text']
        day3_day=day3['day']
        day4=root[0][12][11].attrib
        day4_day=day4['day']
        day4_high=day4['high']
        day4_low=day4['low']
        day4_date=day4['date']
        day4_condition=day4['text']
        day4_day=day4['day']
        days=[day1_day,day2_day,day3_day,day4_day]
        if 'wind' in weather_type:
            a='calm'
            if speed > 10:
                a='windy'
            while 1:
                choice=easygui.buttonbox(msg='It will be '+a+' today. The wind is going at '+speed+' mph. It is '+chill+' degrees with wind chill',choices=
                                         ('OK','Weather','Forecast'))
                if choice=='OK':
                    break
                if choice=='Weather':
                    back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
    it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
    \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Wind'))
                    if back=='OK':
                        break
                if choice=='Forecast':
                    back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                   +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                   +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                   +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                   +'\t'+day4_low,choices=('OK','Wind'))
                    if back=='OK':
                        break
            
            
        if 'question' in weather_type:
            hot=[('shorts','hot'),('t-shirt','hot'),('shirt','hot'),('light','hot'),('loose','hot'),('tank top','hot'),('hot','hot'),('beach','hot'),
                 ('suntan lotion','hot'),('bathing suit','hot'),('bikini','hot'),('flip-flop','hot'),('sandals','hot'),
                 ('sunny','hot'),('clear','hot'),('dusty','hot'),('haze','hot')]
            cold=[('coat','cold'),('jacket','cold'),('gloves','cold'),('mittens','cold'),('cold','cold'),('sweater','cold'),('hoodie','cold'),
                  ('mufflers','cold'),('thermal','cold'),('socks','cold'),('boots','cold'),('jeans','cold'),
                  ('pants','cold'),('snow','cold'),('hail','cold'),('sleet','cold')]
            storm=[('umbrella','storm'),('cloudy','storm'),('storm','storm'),('hurricane','storm'),('tornado','storm'),('thunder','storm'),('rain','storm')]
            weather_hot=[('sunny','weather_hot'),('hot','weather_hot'),('clear','weather_hot'),
                         ('fair','weather_hot'),('partly cloudy','weather_hot'),('smoky','weather_hot'),
                         ('dust','weather_hot'),('haze','weather_hot'),('sunny','weather_hot'),
                         ('hot','weather_hot'),('clear','weather_hot'),('fair','weather_hot'),
                         ('partly cloudy','weather_hot'),('fair','weather_hot'),('clear','weather_hot'),
                         ('clear','weather_hot'),('clear','weather_hot'),('AM Clouds/PM Sun','weather_hot')]
            stormy=[('tornado','stormy'),('tropical storm','stormy'),('hurricane','stormy'),
                    ('thunderstorm','stormy'),('thundershowers','stormy'),('rain','stormy'),('cloudy','stormy'),
                    ('mostly cloudy','stormy'),('few showers','stormy'),('rain','stormy'),('rain','stormy')]
            weather_cold=[('sleet','weather_cold'),
                          ('snow','weather_cold'),('showers','weather_cold'),('freezing','weather_cold'),
                          ('drizzle','weather_cold'),('flurries','weather_cold'),('hail','weather_cold'),
                          ('foggy','weather_cold'),('cold','weather_cold'),('blustery','weather_cold'),
                          ('windy','weather_cold'),('sleet','weather_cold'),
                          ('snow','weather_cold'),('showers','weather_cold')]
            user_q=[ ]
            for (words,sentiment) in storm+cold+hot:
                words_filtered=[e.lower() for e in words.split() if len(e)>=3]
                user_q.append((words_filtered,sentiment))
            word_features=get_features(get_words(user_q))
            trainer=nltk.classify.apply_features(extract_features,user_q)
            classer=nltk.NaiveBayesClassifier.train(trainer)
            user_type=classer.classify(extract_features(i.split()))
            if 'week' in i:
                conditions=[day1_condition,day2_condition,day3_condition,day4_condition]
                for u in conditions:
                    for f in weather_hot:
                        if f[0].lower()==u.lower():
                            condition_type=f[1]
                if condition_type=='None':
                    for f in weather_cold:
                        if t[0].lower()==u.lower():
                            condition_type=f[1]
                if condition_type=='None':
                    for f in stormy:
                        if f[0].lower()==u.lower():
                            condition_type=f[1]
                if user_type=='hot' and condition_type=='stormy':
                    while 1:
                        os.system("say There's going to be bad weather!")
                        choice=easygui.buttonbox('There\'s going to be bad weather!\nCondition: '+day_condition,
                                                 choices=('Weather','Forecast','OK'))
                        if choice=='OK':
                            break
                        if choice=='Weather':
                            back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
            it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
            \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                               '\nSunrise:'+sunrise,choices=('OK','Condition'))
                            if back=='OK':
                                break
                        if choice=='Forecast':
                            back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                           +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                           +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                           +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                           +'\t'+day4_low,choices=('OK','Condition'))
                            if back=='OK':
                                break
                if user_type=='stormy' and condtion_type=='stormy':
                    while 1:
                        choice=easygui.buttonbox('Yes, the weather will not be good today!\nCondition: '+day_condition,
                                                 choices=('Weather','Forecast','OK'))
                        if choice=='OK':
                            break
                        if choice=='Weather':
                            back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
            it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
            \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                               '\nSunrise:'+sunrise,choices=('OK','Condition'))
                            if back=='OK':
                                break
                        if choice=='Forecast':
                            back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                           +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                           +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                           +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                           +'\t'+day4_low,choices=('OK','Condition'))
                            if back=='OK':
                                break
                if user_type=='cold' and condition_type=='stormy':
                    while 1:
                        choice=easygui.buttonbox('The weather is not going to be good!\nCondition: '+day_condition,
                                                 choices=('Weather','Forecast','OK'))
                        if choice=='OK':
                            break
                        if choice=='Weather':
                            back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
            it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
            \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                               '\nSunrise:'+sunrise,choices=('OK','Condition'))
                            if back=='OK':
                                break
                        if choice=='Forecast':
                            back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                           +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                           +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                           +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                           +'\t'+day4_low,choices=('OK','Condition'))
                            if back=='OK':
                                break
                    
                if user_type=='hot' and condition_type=='weather_hot':
                    while 1:
                        choice=easygui.buttonbox('Yes, its going to be warm today!\nCondition: '+day_condition,
                                                 choices=('Weather','Forecast','OK'))
                        if choice=='OK':
                            break
                        if choice=='Weather':
                            back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
            it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
            \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                               '\nSunrise:'+sunrise,choices=('OK','Condition'))
                            if back=='OK':
                                break
                        if choice=='Forecast':
                            back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                           +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                           +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                           +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                           +'\t'+day4_low,choices=('OK','Condition'))
                            if back=='OK':
                                break
                if user_type=='hot' and condition_type=='weather_cold':
                    while 1:
                        choice=easygui.buttonbox('No, dress up its going to be cold!\nCondition: '+day_condition\
                                              ,choices=('Weather','Forecast','OK'))
                        if choice=='OK':
                            break
                        if choice=='Weather':
                            back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
            it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
            \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                               '\nSunrise:'+sunrise,choices=('OK','Condition'))
                            if back=='OK':
                                break
                        if choice=='Forecast':
                            back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                           +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                           +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                           +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                           +'\t'+day4_low,choices=('OK','Condition'))
                            if back=='OK':
                                break
                if user_type=='cold' and condition_type=='weather_cold':
                    while 1:
                        choice=easygui.buttonbox('Yes, dress up its going to be cold!\nCondition: '+day_condition\
                                              ,choices=('Weather','Forecast','OK'))
                        if choice=='OK':
                            break
                        if choice=='Weather':
                            back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
            it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
            \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                               '\nSunrise:'+sunrise,choices=('OK','Condition'))
                            if back=='OK':
                                break
                        if choice=='Forecast':
                            back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                           +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                           +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                           +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                           +'\t'+day4_low,choices=('OK','Condition'))
                            if back=='OK':
                                break
                if user_type=='cold' and condition_type=='weather_hot':
                    while 1:
                        choice=easygui.buttonbox('No, the weather will be good today!\nCondition: '+day_condition\
                                              ,choices=('Weather','Forecast','OK'))
                        if choice=='OK':
                            break
                        if choice=='Weather':
                            back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
            it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
            \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                               '\nSunrise:'+sunrise,choices=('OK','Condition'))
                            if back=='OK':
                                break
                        if choice=='Forecast':
                            back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                           +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                           +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                           +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                           +'\t'+day4_low,choices=('OK','Condition'))
                        
                    if user_type=='storm' and condition_type=='weather_hot':
                        choice=easygui.buttonbox('No, its going to be hot today!\nCondition: '+day_condition\
                                              ,choices=('Weather','Forecast','OK'))
                        if choice=='OK':
                            break
                        if choice=='Weather':
                            back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
            it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
            \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                               '\nSunrise:'+sunrise,choices=('OK','Condition'))
                            if back=='OK':
                                break
                        if choice=='Forecast':
                            back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                           +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                           +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                           +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                           +'\t'+day4_low,choices=('OK','Condition'))
                    
                
                    
                if user_type=='storm' and condition_type=='weather_cold':
                    while 1:
                        choice=easygui.buttonbox('No, its going to be cold today!\nCondition: '+day_condition\
                                                  ,choices=('Weather','Forecast','OK'))
                        if choice=='OK':
                            break
                        if choice=='Weather':
                            back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
                it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
                \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                               '\nSunrise:'+sunrise,choices=('OK','Condition'))
                            if back=='OK':
                                break
                            if choice=='Forecast':
                                back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                               +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                               +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                               +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                               +'\t'+day4_low,choices=('OK','Condition'))
                    
            condition_type='None'
            r='true'
            w='true'
            if 'today' in i:
                w='false'
            if 'day' in i and w=='true':
                for w in i.split():
                    try:
                        r=int(w)
                    except:
                        pass
                if 'friday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Fri':
                            break
                if 'thursday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Thu':
                            break
                if 'wednesday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Wed':
                            break
                if 'saturday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Sat':
                            break
                if 'sunday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Sun':
                            break
                if 'tuesday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Tue':
                            break
                if 'monday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Mon':
                            break
                if week[str(r)]=='day1_condition':
                    day_condition=day1_condition
                    high=day2_high
                    low=day2_low
                if week[str(r)]=='day2_condition':
                    day_condition=day2_condition
                    high=day2_high
                    low=day2_low
                if week[str(r)]=='day3_condition':
                    day_condition=day3_condition
                    high=day3_high
                    low=day3_low
                if week[str(r)]=='day4_condition':
                    day_condition=day4_condition
                    high=day4_high
                    low=day4_low
                    
            for t in weather_hot:
                if t[0].lower()==day_condition.lower():
                    easygui.msgbox(t)
                    condition_type=t[1]
            if condition_type=='None':
                for t in weather_cold:
                    if t[0].lower()==day_condition.lower():
                        condition_type=t[1]
            if condition_type=='None':
                for t in stormy:
                    if t[0].lower()==day_condition.lower():
                        condition_type=t[1]
                
            easygui.msgbox(user_type+'\n'+condition_type)
            t=current['temp']
            if condition_type!='stormy':
                if int(low)<40:
                    condition_type='weather_cold'
                if int(t)<50:
                    condition_type='weather_cold'
                if int(high)>70:
                    condition_type='weather_hot'
            if user_type=='hot' and condition_type=='stormy':
                os.system('say There is going to be bad weather')
                while 1:
                    choice=easygui.buttonbox('There\'s going to be bad weather!\nCondition: '+day_condition,
                                             choices=('Weather','Forecast','OK'))
                    if choice=='OK':
                        break
                    if choice=='Weather':
                        back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
        it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
        \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Condition'))
                        if back=='OK':
                            break
                    if choice=='Forecast':
                        back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                       +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                       +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                       +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                       +'\t'+day4_low,choices=('OK','Condition'))
                        if back=='OK':
                            break
            if user_type=='stormy' and condtion_type=='stormy':
                os.system('say Yes, the weather will not be good today')
                while 1:
                    choice=easygui.buttonbox('Yes, the weather will not be good today!\nCondition: '+day_condition,
                                             choices=('Weather','Forecast','OK'))
                    if choice=='OK':
                        break
                    if choice=='Weather':
                        back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
        it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
        \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Condition'))
                        if back=='OK':
                            break
                    if choice=='Forecast':
                        back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                       +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                       +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                       +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                       +'\t'+day4_low,choices=('OK','Condition'))
                        if back=='OK':
                            break
            if user_type=='cold' and condition_type=='stormy':
                os.system('say The weather is not going to be good')
                while 1:
                    choice=easygui.buttonbox('The weather is not going to be good!\nCondition: '+day_condition,
                                             choices=('Weather','Forecast','OK'))
                    if choice=='OK':
                        break
                    if choice=='Weather':
                        back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
        it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
        \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Condition'))
                        if back=='OK':
                            break
                    if choice=='Forecast':
                        back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                       +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                       +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                       +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                       +'\t'+day4_low,choices=('OK','Condition'))
                        if back=='OK':
                            break
                
            if user_type=='hot' and condition_type=='weather_hot':
                os.system('say Yes, its going to be warm today')
                while 1:
                    choice=easygui.buttonbox('Yes, its going to be warm today!\nCondition: '+day_condition,
                                             choices=('Weather','Forecast','OK'))
                    if choice=='OK':
                        break
                    if choice=='Weather':
                        back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
        it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
        \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Condition'))
                        if back=='OK':
                            break
                    if choice=='Forecast':
                        back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                       +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                       +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                       +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                       +'\t'+day4_low,choices=('OK','Condition'))
                        if back=='OK':
                            break
            if user_type=='hot' and condition_type=='weather_cold':
                os.system('say No, dress up its going to be cold')
                while 1:
                    choice=easygui.buttonbox('No, dress up its going to be cold!\nCondition: '+day_condition\
                                          ,choices=('Weather','Forecast','OK'))
                    if choice=='OK':
                        break
                    if choice=='Weather':
                        back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
        it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
        \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Condition'))
                        if back=='OK':
                            break
                    if choice=='Forecast':
                        back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                       +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                       +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                       +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                       +'\t'+day4_low,choices=('OK','Condition'))
                        if back=='OK':
                            break
            if user_type=='cold' and condition_type=='weather_cold':
                os.system('say Yes, dress up its going to be cold')
                while 1:
                    choice=easygui.buttonbox('Yes, dress up its going to be cold!\nCondition: '+day_condition\
                                          ,choices=('Weather','Forecast','OK'))
                    if choice=='OK':
                        break
                    if choice=='Weather':
                        back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
        it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
        \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Condition'))
                        if back=='OK':
                            break
                    if choice=='Forecast':
                        back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                       +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                       +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                       +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                       +'\t'+day4_low,choices=('OK','Condition'))
                        if back=='OK':
                            break
            if user_type=='cold' and condition_type=='weather_hot':
                os.system('say No, the weather will be good today')
                while 1:
                    choice=easygui.buttonbox('No, the weather will be good today!\nCondition: '+day_condition\
                                          ,choices=('Weather','Forecast','OK'))
                    if choice=='OK':
                        break
                    if choice=='Weather':
                        back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
        it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
        \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Condition'))
                        if back=='OK':
                            break
                    if choice=='Forecast':
                        back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                       +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                       +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                       +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                       +'\t'+day4_low,choices=('OK','Condition'))
                    
                if user_type=='storm' and condition_type=='weather_hot':
                    os.system('say No, its going to be hot today')
                    choice=easygui.buttonbox('No, its going to be hot today!\nCondition: '+day_condition\
                                          ,choices=('Weather','Forecast','OK'))
                    if choice=='OK':
                        break
                    if choice=='Weather':
                        back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
        it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
        \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Condition'))
                        if back=='OK':
                            break
                    if choice=='Forecast':
                        back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                       +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                       +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                       +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                       +'\t'+day4_low,choices=('OK','Condition'))
                
            
                
            if user_type=='storm' and condition_type=='weather_cold':
                os.system('say No, its going to be cold today') 
                while 1:
                    choice=easygui.buttonbox('No, its going to be cold today!\nCondition: '+day_condition\
                                              ,choices=('Weather','Forecast','OK'))
                    if choice=='OK':
                        break
                    if choice=='Weather':
                        back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
            it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
            \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Condition'))
                        if back=='OK':
                            break
                        if choice=='Forecast':
                            back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                           +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                           +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                           +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                           +'\t'+day4_low,choices=('OK','Condition'))
                  
        if 'temp' in weather_type:
            r='true'
            w='true'
            if 'today' in i:
                w='false'
            if 'day' in i and w=='true':
                for w in i.split():
                    try:
                        r=int(w)
                    except:
                        pass
                if 'friday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Fri':
                            break
                if 'thursday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Thu':
                            break
                if 'wednesday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Wed':
                            break
                if 'saturday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Sat':
                            break
                if 'sunday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Sun':
                            break
                if 'tuesday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Tue':
                            break
                if 'monday' in i:
                    r=0
                    for h in days:
                        r=r+1
                        if h=='Mon':
                            break
                if week[str(r)]=='day1_condition':
                    day=day1_day
                    high=day2_high
                    low=day2_low
                if week[str(r)]=='day2_condition':
                    day=day2_day
                    high=day2_high
                    low=day2_low
                if week[str(r)]=='day3_condition':
                    day=day3_day
                    high=day3_high
                    low=day3_low
                if week[str(r)]=='day4_condition':
                    day=day4_day
                    high=day4_high
                    low=day4_low
            while 1:
                os.system('say The high is '+high+' degrees today')
                choice=easygui.buttonbox(msg='Temperature for '+day+'\nHigh: '+high+' degrees\nLow: '+low+' degrees\nCurrent: '+t+' degrees',choices=
                                         ('OK','Weather','Forecast'))
                if choice=='OK':
                    break
                if choice=='Weather':
                    back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
    it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
    \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,choices=('OK','Temperature'))
                    if back=='OK':
                        break
                if choice=='Forecast':
                    back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                   +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                   +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                   +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                   +'\t'+day4_low,choices=('OK','Temperature'))
                    if back=='OK':
                        break

        if 'general' in weather_type:
            os.system('say Today it will be'+day_condition)
            while 1:
                choice=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
    it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\
    \nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,
                                         choices=('OK','Forecast'))
                if choice=='Forecast':
                    back=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                   +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                   +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                   +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                   +'\t'+day4_low,choices=('OK','Weather'))
                    if back=='OK':
                        break
                if choice=='OK':
                    break
        if 'forecast' in weather_type:
            os.system("say Here is the forecast")
            while 1:
                choice=easygui.buttonbox(msg=day1_day+', '+day1_date+'\t'+day1_condition+'\t'+day1_high\
                                   +'\t'+day1_low+'\n'+day2_day+', '+day2_date+'\t'+day2_condition+'\t'+day2_high\
                                   +'\t'+day2_low+'\n'+day3_day+', '+day3_date+'\t'+day3_condition+'\t'+day3_high\
                                   +'\t'+day3_low+'\n'+day4_day+', '+day4_date+'\t'+day4_condition+'\t'+day4_high\
                                   +'\t'+day4_low,choices=('OK','Weather'))
                if choice=='OK':
                    break
                if choice=='Weather':
                    back=easygui.buttonbox(msg=conditions+'\nHigh is '+high+' degrees and the low is '+low+' degrees\nToday\
        it will be '+day_condition+'\nRight now it is '+current_condition+' and '+t+' degrees\nThe wind is going at '+speed+' and it is '+chill+' degrees with wind chill\nSunset: '+sunset+\
                                           '\nSunrise:'+sunrise,
                                             choices=('OK','Forecast'))
                    if back=="OK":
                        break
                
                
        
        
                
    
    
    
              
