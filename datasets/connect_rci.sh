#!/bin/bash

#killall sshfs
#fusermount -u mnt/rci

sshfs -o reconnect,ServerAliveInterval=5,nonempty,password_stdin,allow_other 
pokokrsi1@login3.rci.cvut.cz:/home/pokorsi1/data/ ~/data/ <<< "pasword"
