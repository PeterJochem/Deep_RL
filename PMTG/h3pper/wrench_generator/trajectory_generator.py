import glob
import csv
import numpy as np
import bisect


class trajectory_interpolator:
    """
    A class for interpolating between existing trajectories.
    (and possibly, later, generating new trajectories.)
    ...
    Attributes:
            trajectories: numpy.ndarray
                    a multidimensional NumPy array containing MATLAB-generated
                    trajectories for a variety of running velocities ("vel") and a
                    variety of ground stiffnesses ("gnd_stiffness", defauls to 1).
            params_dict: dict
                    a dictionary that helps
                    trajectory_interpolator.interpolate_trajectories() access the
                    correct slices of trajectory_interpolator.trajectories.
    Methods:
            interpolate_trajectories(gnd_stiffness=1.0,vel)
                    interpolates between slices of trajectory_interpolator.trajectories
                    and returns an estimated trajectory for the provided values of
                    "gnd_stiffness" (defaults to 1) and "vel".
            get_state_control_time(vel,gnd_stiffness=1.0,t)
                    gets a [state,control,time] list from a trajectory,


    1. The primary methods in this class are
            trajectory_interpolator(gnd_stiffness,vel,time);
            get_state_control_time(t,vel,gnd_stiffness)
    2. This class also contains an __init__() that runs when the class is
            instantiated.
    3. The __init__() method calls another method, parse_csvs(),
            that parses a collection of CSV files and returns a dictionary.
            This dictionary is then stored as an attribute of the class.
            This attribute is accessible to the user through trajectory_interpolator().
    """
    params_dict = {
        'gnd_stiffnesses': [],
        'vels': []}
    trajectories = []

    # The following functions for searching sorted lists are copied & pasted,
    #	shamelessly, from https://docs.python.org/3.5/library/bisect.html#searching-sorted-lists
    def __index(self, a, x):
        'Locate the leftmost value exactly equal to x'
        i = bisect.bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        raise ValueError

    def __find_lt(self, a, x):
        'Find rightmost value less than x'
        i = bisect.bisect_left(a, x)
        if i:
            return a[i-1]
        raise ValueError

    def __find_le(self, a, x):
        'Find rightmost value less than or equal to x'
        i = bisect.bisect_right(a, x)
        if i:
            return a[i-1]
        raise ValueError

    def __find_gt(self, a, x):
        'Find leftmost value greater than x'
        i = bisect.bisect_right(a, x)
        if i != len(a):
            return a[i]
        raise ValueError

    def __find_ge(self, a, x):
        'Find leftmost item greater than or equal to x'
        i = bisect.bisect_left(a, x)
        if i != len(a):
            return a[i]
        raise ValueError

    def __interp_between_2darrays(self, arr1, arr2, x1, x2, x):
        """ Linearly interpolates between two 2D arrays and returns a new array.

        The idea is that each 2D array is parameterized by a single variable,
        such as running speed. This method can be used when asking "what is the
        trajectory for a running speed for which I don't have raw data, but
        which is between two running speeds for which I _do_ have raw data?"

        Note: I don't know if there is a NumPy or SciPy method for doing this;
        rather than dig around forever, I just wrote this method.
        Feel free to revise/replace it.
        """

        allowed_types = [int, float, np.int32,
                         np.int64, np.float32, np.float64]

        assert (type(arr1) is np.ndarray)
        assert (type(arr2) is np.ndarray)
        assert (arr1.shape == arr2.shape)
        assert (type(x1) in allowed_types)
        assert (type(x2) in allowed_types)
        assert (type(x) in allowed_types)
        #assert(type(arr1[0,0]) in allowed_types)
        #assert(type(arr2[0,0]) in allowed_types)

        # handle edge cases:
        if x == x1:  # query value equals lower bound
            return arr1
        elif x == x2:  # query value equals upper bound
            return arr2
        else:  # query value between lower and upper bounds
            return arr1*(1 - ((x-x1)/(x2-x1))) + arr2*((x - x1)/(x2-x1))

    def __init__(self, csv_dir):
        """
        Runs when a trajectory_generator class is instantiated.
        This method searches recursively through "csv_dir",looking for CSV files.
        Once it has found some CSV files, it parses their contents, into two
        attributes:
                trajectory_generator.trajectories: numpy.ndarray
                        a multidimensional NumPy array containing MATLAB-generated
                        trajectories for a variety of running velocities ("vel") and a
                        variety of ground stiffnesses ("gnd_stiffness", defauls to 1).
                trajectory_generator.params_dict: dict
                        a dictionary that helps
                        trajectory_interpolator.interpolate_trajectories() access the
                        correct slices of trajectory_interpolator.trajectories.
        """
        # glob.glob returns a list in the order in which entries appear in the
        # filesystem, so it might appear random; sort it:
        pathnames = sorted(glob.glob((csv_dir + '**/*.csv'), recursive=True))
        # print(pathnames)
        # TODO: input checking on csv_dir, error handling if no CSVs found.
        for pathname in pathnames:
            with open(pathname, 'r') as f:
                lines = f.readlines()
                # skipping 1st line which just says "header"
                header = lines[1:4]

                # parse gait parameters from header:
                # (this approach assumes the header has a very specific
                # structure, so it's brittle and could be robustified)
                v = float(header[0][2:])
                T = float(header[1][2:])
                gnd_stiffness = float(header[2][5:])

                if gnd_stiffness not in self.params_dict['gnd_stiffnesses']:
                    self.params_dict['gnd_stiffnesses'].append(gnd_stiffness)
                if gnd_stiffness not in self.params_dict['vels']:
                    self.params_dict['vels'].append(v)
                # print('v=',v)
                # print('T=',T)
                #print('zeta =',ground_hardness)

                # parse gait trajectory (state, control, and time) from body:
                body = lines[5:]
                # print(len(body))
                body_list = list(csv.reader(body, delimiter=','))
                body_header = body_list[0]  # first row has column names
                body_body = body_list[1:]  # remaining rows have data
                #print('body_header =',body_header)
                ncols = len(body_header)
                nrows = len(body_body)
                #print('ncols =',ncols)
                #print('nrows =',nrows)
                traj_slice = np.zeros([nrows, ncols])
                # print(traj_slice.shape)

                for row_num in range(nrows):
                    for col_num in range(ncols):
                        traj_slice[row_num, col_num] = float(
                            body_body[row_num][col_num])
                self.trajectories.append(traj_slice)
        self.trajectories = np.array(self.trajectories)
        # print(self.trajectories)

    def interpolate_trajectories(self, vel, gnd_stiffness=1.0):
        """
        Interpolate between two existing trajectories.
        Each existing trajectory is a NumPy array and is associated with a
        particular "vel" value and (later, not supported yet) a particular
        "gnd_stiffness" value.
        """

        allowed_types = [int, float, np.int32,
                         np.int64, np.float32, np.float64]

        assert (type(vel) in allowed_types)
        assert (type(gnd_stiffness) in allowed_types)

        # The following block of code can be duplicated for handling
        # user-defined gnd_stiffness values:
        out_of_bounds_cond1 = vel > max(self.params_dict['vels'])
        out_of_bounds_cond2 = vel < min(self.params_dict['vels'])
        if out_of_bounds_cond1 or out_of_bounds_cond2:
            print('vel =', vel, ' is outside the range spanned by the dataset.')
            return 0
        nearest_le_vel = self.__find_le(self.params_dict['vels'], vel)
        nearest_le_index = self.__index(
            self.params_dict['vels'], nearest_le_vel)
        nearest_ge_vel = self.__find_ge(self.params_dict['vels'], vel)
        nearest_ge_index = self.__index(
            self.params_dict['vels'], nearest_ge_vel)

        # print(nearest_le_vel,'<=',vel,'<=',nearest_ge_vel)
        #print("params_dict['vels'][",nearest_le_index,'] =',nearest_le_vel)
        #print("params_dict['vels'][",nearest_ge_index,'] =',nearest_ge_vel)

        traj_le = self.trajectories[nearest_le_index]
        traj_ge = self.trajectories[nearest_ge_index]

        return self.__interp_between_2darrays(traj_le,
                                              traj_ge,
                                              nearest_le_vel,
                                              nearest_ge_vel,
                                              vel)

    def get_state_control_time(self, t, vel, gnd_stiffness=1.0):
        """
        Get a [time,state,control] list from a trajectory,
        given some elapsed time "t" into a gait, a running speed ("vel"), 
        and a ground stiffness ("gnd_stiffness"),
        Each existing trajectory is a NumPy array and is associated with a
        particular "vel" value and (later, not supported yet) a particular
        "gnd_stiffness" value.
        This method will call trajectory_generator.interpolate_trajectories() if
        necessary (i.e., if trajectory_generator.trajectories 
        and trajectory_generator.params_dict don't contain an
        entry for the supplied values of "vel" and "gnd_stiffness").
        Because this method will call
        trajectory_generator.interpolate_trajectories() if the
        [vel,gnd_stiffness] pair doesn't exist yet, it is inefficient to call
        this method repeatedly, using the same [vel,gnd_stiffness] while varying
        t.
        """

        allowed_types = [int, float, np.int32,
                         np.int64, np.float32, np.float64]

        assert (type(vel) in allowed_types)
        assert (type(gnd_stiffness) in allowed_types)
        assert (type(t) in allowed_types)

        # Does self.trajectories already have an entry for (gnd_stiffness,vel)?
        if gnd_stiffness in self.params_dict['gnd_stiffnesses']:
            out_of_bounds_cond1 = vel > max(self.params_dict['vels'])
            out_of_bounds_cond2 = vel < min(self.params_dict['vels'])
            if vel in self.params_dict['vels']:
                pass
                # print('vel =',vel,' is already in the dataset.')
            else:
                # print('vel =',vel,' is not already in the dataset.')
                if out_of_bounds_cond1 or out_of_bounds_cond2:
                    print('vel =', vel,
                          ' is outside the range spanned by the dataset.')
                    return
            new_traj = self.interpolate_trajectories(vel, gnd_stiffness)
            new_t = new_traj[:, -1]  # we'll need this in the next step
        else:
            print('varying the gnd_stiffness is not supported yet.')
            return

        # Find the times that bound "t" above and below:
        nearest_le_t = self.__find_le(new_t, t)
        nearest_le_index = self.__index(new_t, nearest_le_t)
        nearest_ge_t = self.__find_ge(new_t, t)
        nearest_ge_index = self.__index(new_t, nearest_ge_t)

        # Linearly interpolate between the two rows,
        #	using self.__interp_between_2darrays():
        return self.__interp_between_2darrays(new_traj[nearest_le_index],
                                              new_traj[nearest_ge_index],
                                              nearest_le_t,
                                              nearest_ge_t,
                                              t)


    """Description"""
    def get_trajectories(self, vel, numPoints=10, gnd_stiffness=1.0):
                
        wrench_series = [] # np.zeros((numPoints, 6)) # [u_theta, 0.0, 0.0, 0.0, u_x, u_y]
            
        traj = self.interpolate_trajectories(vel, gnd_stiffness) # (101, 16) 
        initial_conditions = traj[0][:12]
        minTime = traj[0][-1] 
        maxTime = traj[-1][-1]

        idx = np.round(np.linspace(0, len(traj) - 1, numPoints)).astype(int)
        downsampled_traj = traj[idx]

        for i in range(len(downsampled_traj)):
    
                nextWrench = np.zeros(6)
                nextWrench[0] = downsampled_traj[i][14]
                nextWrench[4] = downsampled_traj[i][13]
                nextWrench[5] = downsampled_traj[i][12]
                
                wrench_series.append(nextWrench)
       
        wrench_series = np.array(wrench_series)
        return minTime, maxTime, initial_conditions, wrench_series
        

# some driver code:
if __name__ == "__main__":
    from trajectory_generator import trajectory_interpolator
    from matplotlib import pyplot as plt

    ti = trajectory_interpolator('data/CSVs/zeta_1.0/')
    print(ti.__doc__)

    old_vel = 0.3
    new_vel = 0.35
    gnd_stiffness = 1.0
    t = 0.1
    old_tsc = ti.get_state_control_time(t, old_vel, gnd_stiffness)
    print('time-state-control for t = %f, vel = %f:' % (t, old_vel))
    print(old_tsc)

    new_tsc = ti.get_state_control_time(t, new_vel, gnd_stiffness)
    print('time-state-control for t = %f, vel = %f:' % (t, new_vel))
    print(new_tsc)

    bad_tsc1 = ti.get_state_control_time(t, 0.2, gnd_stiffness)
    bad_tsc2 = ti.get_state_control_time(t, 0.35, gnd_stiffness=0.9)

    # Just checking on the dataset:
    #trajs = ti.trajectories
    #print('type(trajs) =',type(trajs))
    #print('trajs[0] =',trajs[0])
    #print('trajs[0].shape =',trajs[0].shape)
    #print('trajs.shape =',trajs.shape)
    # print(ti.params_dict)

    new_traj = ti.interpolate_trajectories(0.35)
    print("The new trajectory is \n\n\n\n")
    print(new_traj)
    print(new_traj.shape)
    print("\n\n\n\n")
    #print('new_traj.shape =',new_traj.shape)
    new_t = new_traj[:, -1]
    #print('The new trajectory takes',new_t[-1],'seconds.')

    lb_traj = ti.interpolate_trajectories(0.3)
    ub_traj = ti.interpolate_trajectories(0.4)

    lb_t = lb_traj[:, -1]
    ub_t = ub_traj[:, -1]

    # Let's visualize a few slices of the interpolated trajectory:
    fig, axs = plt.subplots(3, 1, sharex=True)

    # First, compare the vertical position of the body ("y_b"):
    axs[0].plot(new_traj[:, 2], label='new y_b')
    axs[0].plot(lb_traj[:, 2], label='lb y_b')
    axs[0].plot(ub_traj[:, 2], label='ub y_b')
    axs[0].legend(loc='best')
    axs[0].set(ylabel='position [m]',
               title='interpolating between two trajectories')

    # Second, compare the vertical control effort ("u_y")
    axs[1].plot(new_traj[:, 13], label='new u_y')
    axs[1].plot(lb_traj[:, 13], label='lb u_y')
    axs[1].plot(ub_traj[:, 13], label='ub u_y')
    axs[1].legend(loc='best')
    axs[1].set(ylabel='force [N]')

    # Third, compare the time for each trajectory ("t")
    axs[2].plot(new_t, label='new t')
    axs[2].plot(lb_t, label='lb t')
    axs[2].plot(ub_t, label='ub t')
    axs[2].legend(loc='best')
    axs[2].set(xlabel='trajectory step number', ylabel='time [s]')

    plt.show()
