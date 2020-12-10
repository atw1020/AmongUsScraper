# Game Classifier

The Game Classifies a given image of as one of 
5 categories: Lobby, Gameplay, Meeting, Over and
Other. It is a simple computer vision problem
and is solved using a convolutional neural
network

### Case 1: Lobby

An image classified as "Lobby" is an among us
Lobby where gameplay has not begun

![Case 1: Lobby](Resources/Case%201:%20Lobby.jpg)

###Case 2: Gameplay

An image Classified as "gameplay" is an among us
game that is currently in progress

![Case 2: Gameplay](Resources/Case%202:%20Gameplay.jpg)

###Case 3: Meeting

An image classified as "Meeting" shows an among
us game where the players are in a meeting

![Case 3: Meeting](Resources/Case%203:%20Meeting.jpg)

###Case 4: Over

An image classified as "Over" represents the
final screen of an among us game

![Case 4: Over](Resources/Case%204:%20Over.jpg)

###Case 5: Other

Any Image that does not fall into one of the
above categories (ie. gameplay of another game)
is classified as "Other"

![Case 5: Other](Resources/Case%205:%20Other.png)

# Winner Identifier

The Winner Identifier Is a Neural Network that
given an ending screen identifies the colors of each of the winning players. It can also
be thought of as an image classifier where it
outputs all of the colors of the players that
won.

For example, in the image below, The Neural
network would output "Purple, Red, Blue, Black, Orange, Yellow, White, Green Pink"
Corresponding to the Colors that won the game

![Case 4: Over](Resources/Case%204:%20Over.jpg)

# Player Identifier

The Player Identifier Is a more complicated
Neural Network that Takes in an image of a
meeting screen and returns the names of all 
the participating players. Since the winning
screen does not always show the names of the
colors, it is necessary to create a map from
Colors to the names of the players in question
This is done by reading the text on the image
and outputting a string of variable length for
the username of each player. This is done using
Recurrent Neural Networks and is not a simple
image classifier like the other Neural Networks

For Example, in the Image below, The Network
Should return Something similar to the
following table

|Color|Name|
|---|---|
|Lime|Germaine|
|Cyan|Dubs|
|Black|KEI|
|Blue|Blaaczek|
|Pink|Aliza|
|Yellow|SuckMighty|
|Green|Green|
|Red|Kirito|
|Purple|Michelle|
|White|bb|

![Case 3: Meeting](Resources/Case%203:%20Meeting.jpg)