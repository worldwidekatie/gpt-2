"""This is just to reshape the chatbot data into a .txt format"""

import json

# with open('chatbot/data.json') as f:
#   data = json.load(f)


# text = ""
# count = 0
# for i in data:
#     if count < 100:
#         sub_count = 0
#         for x in data[i]['log']:
#             if sub_count % 2 == 0:
#                 text += "Human: " + x['text'] + "\r\n"
#                 sub_count +=1 
#             else:
#                 text += "AI: " + x['text'] + "\r\n"
#                 sub_count += 1
#         count +=1
#     else:
#         break


# tfile = open('chatdata_100.txt', 'a')
# tfile.write(text)
# tfile.close()


characters = {}
with open('movie-chat/movie_characters_metadata.txt', 'r', encoding='iso-8859-1') as chars:
  for char in chars:
      characters[char.split(' +++$+++ ')[0]] = char.split(' +++$+++ ')[1] + ": "

lines = {}
with open('movie-chat/movie_lines.txt', 'r', encoding='iso-8859-1') as f:
    for line in f.readlines():
        lines[line.split(' +++$+++ ')[0]] = characters[line.split(' +++$+++ ')[1]] + line.split(' +++$+++ ')[-1]
        
dialogues = ""
with open('movie-chat/movie_conversations.txt') as convos:
  for convo in convos:
      ls = convo.split(' +++$+++ ')[3].replace('\n', "").replace('[', "").replace("]", "").replace("'", "").split(", ")
      for l in ls:
          dialogues += lines[l]

tfile = open('dialogues.txt', 'a')
tfile.write(dialogues)
tfile.close()

# text = ""
# count = 0
# for i in data:
#     if count < 100:
#         sub_count = 0
#         for x in data[i]['log']:
#             if sub_count % 2 == 0:
#                 text += "Human: " + x['text'] + "\r\n"
#                 sub_count +=1 
#             else:
#                 text += "AI: " + x['text'] + "\r\n"
#                 sub_count += 1
#         count +=1
#     else:
#         break


# tfile = open('chatdata_100.txt', 'a')
# tfile.write(text)
# tfile.close()