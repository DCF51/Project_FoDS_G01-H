# FoDS_Project_G01-H
Foundations of Data Science - Project of Group G01-H

Folder is organized as follows:
- code #here the code is stored
- data #here the data.csv file is stored
- output #here all our graphs and other outputs are stored

to access these from the code use '../[folder name]/[file name]'

To clone the repository to your own computer:
1. go to the right directory
  
  e.g. cd ~/Documents/FoDS

2. clone online repository to that folder
  
  git clone https://github.com/DCF51/FoDS_Project_G01-H.git

To add changes to github:
1. go to the right directory
  
  e.g. cd ~/Documents/FoDS/FoDS_Project_G01-H

2. check status of the directory (what you modified, what is ready for a commit...)

  git status

3. commit changes

  git add [file name] --> to add a new file

  git commit -f [file name] -m [comment on the commit] --> to change existing file
  
  git commit -a -m "[comment on the commit]" --> to change all files in the directory
  
  you can also do this if you have a longer comment:
  
  git commit -a --> a text window opens --> write your comments on any line without a # in front --> press "ESC" then ":wq" to exit

4. Add the local changes to the online repository

  git push -f origin [branch name]
  
To navigate branches:
1. create a new branch (only if necessary, e.g. we could create a branch for a part of the code that 2 people are working on simultaneously, meaning where there are two completely different versions)

  git branch [branch name] e.g. branch1

2. chose branch/change 'active' branch

  git checkout [branch name]

  [branch name] could be 'main', 'branch1' etc.
