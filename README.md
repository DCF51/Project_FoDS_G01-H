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
4. check status of the directory (what you modified, what is ready for a commit...)
  git status
5. commit changes
  git add [file name] --> to add a new file
  git commit -m [comment on the commit] --> to change existing directory 
6. Add the local changes to the online repository
  git push -f origin [branch name]
  
To navigate branches:
1. create a new branch (only if necessary, e.g. we could create a branch for a part of the code that 2 people are working on simultaneously, meaning where there are two completely different versions)
  git branch [branch name] e.g. branch1
2. chose branch
  git checkout [branch name]
  [branch name] could be 'main', 'branch1' etc.
3. change 'active' branch
  git checkout [branch name]
