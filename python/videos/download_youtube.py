from pytube import YouTube

url = input('>> Prease enter URL : ')
yt = YouTube(url)
for lis in yt.streams.all():
    print(lis)

tag = input('>> Prease enter itag :')
yt.streams.get_by_itag(int(tag)).download(r'/home/gb/Videos/')
