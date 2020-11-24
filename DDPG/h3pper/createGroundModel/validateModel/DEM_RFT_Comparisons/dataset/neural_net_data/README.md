# Description 
This contains the data used to train the neural network

# Format
There are three csvs
```intrude.csv```: Contains data of the foot being driven into the granular material <br />
```extract.csv```: Contains data of the foot moving out of the granular material <br />
```compiledSet.csv```: This is the above two files combined into one <br />

Each csv file has the following format: time (s), gamma (rads), beta (rads), depth (cm), position_x (cm), position_y (cm), position_z (cm), x_dt (cm/s), y_dt (cm/s), z_dt (cm/s), foot angular velocity x (rads/s), foot angular velocity y (rads/s), foot angular velocity z (rads/s), torque x (Nm), torque y (Nm), torque z (Nm), ground reaction force x (N), ground reaction force y (N), ground reaction force z (N)           

