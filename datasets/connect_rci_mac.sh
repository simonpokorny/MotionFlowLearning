!#/bin/bash 

sshfs -o kill_on_unmount,reconnect,allow_other,defer_permissions,direct_io,password_stdin 
pokorsi1@login3.rci.cvut.cz:/home/pokorsi1/data/ ~/mnt/data <<< "password"
