#
#   Author: Wei Pan
#   Copyright   2021
#
#   coarse grains the fine resolution pde solution and saves the data
#
from relative_import_header import *
import sys
from firedrake import *
from firedrake.petsc import PETSc
from utility import Workspace
from firedrake_utility import TorusMeshHierarchy
from tqg.solver import TQGSolver
import numpy as np
from tqg.example2 import TQGExampleTwo as Example2params


desired_spatial_rank = 1
ensemble = ensemble.Ensemble(COMM_WORLD, desired_spatial_rank)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


def pde_data_fname(wspace, file_index, sub_dir=''):
    return wspace.output_name("pde_data_{}".format(file_index), sub_dir) 

def cpde_data_fname(wspace, file_index, sub_dir=''):
    return wspace.output_name("cpde_data_{}".format(file_index), sub_dir)


class CoarseGrainedSolutionGenerator(TQGSolver):

    """ We assume tqg_params is defined using PDE solver settings """
    def __init__(self, tqg_params, comm_manager, workspace, cnx=128, cny=128):
        ensemble_comm = comm_manager.ensemble_comm
        spatial_comm  = comm_manager.comm

        #print(ensemble_comm.size, spatial_comm.size)

        TQGSolver.__init__(self, tqg_params)
        #self.ensemble_size = ensemble_size
        
        self.output_sub_dir = 'cPDESolution'
        #self.output_name = "cpde_data_{}".format(file_id)
        self.file_work_space = workspace 

        if ensemble_comm.rank == 0:
            print("initialiser", flush=True)

        self.cnx=cnx
        self.cny=cny
        self.number_of_obs_per_satellite = 50

        self.cmesh = TorusMeshHierarchy(cnx, cny, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
        self.cfs_cg = FunctionSpace(self.cmesh, "CG", 1)
        self.cfs_dg = FunctionSpace(self.cmesh, "DG", 1)
        self.cvfs = VectorFunctionSpace(self.cmesh, "CG", 1)
        self.ccoords = Function(self.cvfs).interpolate(SpatialCoordinate(self.cmesh))

    def generate_coarse_grained_data_at_obs_sites(self, comm_manager, file_index, dumpfreq):
        spatial_comm = comm_manager.comm
        cmesh = self.cmesh

        cssh = Function(self.cfs_cg, name="SSH")

        data_output_name = cpde_data_fname(self.file_work_space, file_index, self.output_sub_dir)

        with DumbCheckpoint(data_output_name,  mode=FILE_READ, comm=spatial_comm) as data_chk:
            data_chk.load(cssh, name="SSH")

        obs_sites = self.observation_mesh(file_index, self.number_of_obs_per_satellite)
        observations = np.asarray(cssh.at(obs_sites, tolerance=1e-10))
        #print(observations.shape, obs_sites.shape)

        return observations

    def generate_coarse_grained_data_at_all_grid_points(self, comm_manager, file_index, dumpfreq):
        spatial_comm = comm_manager.comm
        cmesh = self.cmesh
        cssh = Function(self.cfs_cg, name="SSH")
        data_output_name = cpde_data_fname(self.file_work_space, file_index, self.output_sub_dir)

        with DumbCheckpoint(data_output_name,  mode=FILE_READ, comm=spatial_comm) as data_chk:
            data_chk.load(cssh, name="SSH")

        #obs_sites = self.observation_mesh(file_index, self.number_of_obs_per_satellite)
        #observations = np.asarray(cssh.at(obs_sites, tolerance=1e-10))
        return cssh.dat.data[:]


    def generate_fine_res_data_at_obs_sites(self, comm_manager, file_index, dumpfreq):
        spatial_comm = comm_manager.comm
        self.load_initial_conditions_from_file(pde_data_fname(self.file_work_space, file_index, "PDESolution"), spatial_comm)
        ssh = assemble(self.psi0 - 0.5 * Function(self.Vcg).project(self.initial_b))

        obs_sites = self.observation_mesh(file_index, self.number_of_obs_per_satellite)
        observations = np.asarray(ssh.at(obs_sites, tolerance=1e-10))

        return observations


    def generate_a_coarse_grained_data(self, comm_manager, file_index, dumpfreq, output_file_visual=None, time_stamp=0.):
        """
        coarse grain and save a fine resolution pde snapshot

        :return: 
        """
        ensemble_comm = comm_manager.ensemble_comm
        spatial_comm  = comm_manager.comm

        PETSc.Sys.Print('generate an coarse grained data',flush=True)
        
        self.load_initial_conditions_from_file(pde_data_fname(self.file_work_space, file_index, "PDESolution"), spatial_comm)

        #pde_data_file_indices = np.arange(1, 1501) # total 1501 saved fine res data files 
        
        # assume q0 is given in initial_cond
        # high res data use spatial parallelism !!!!!
        q0 = Function(self.Vdg, name="PotentialVorticity")
        b0 = Function(self.Vdg, name="Buoyancy")
        self.psi0.rename("Streamfunction")
        #ssh = Function(self.Vcg, name="SSH")  # sea surface height = psi - 0.5 b1
        q0.assign(self.initial_cond)
        b0.assign(self.initial_b)
        Dt = self.Dt
        root = 0
        
        # ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)
        procno_ecomm = ensemble_comm.rank
        procno_scomm = spatial_comm.rank

        ## project down to coarse grid and save data 
        if 1:
            #cmesh = TorusMeshHierarchy(cnx, cny, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
            cq0 = self.coarse_grain_and_project(q0)
            cb0 = self.coarse_grain_and_project(b0)
            cpsi0 = self.coarse_grain_and_project(self.psi0)
            cssh = assemble(cpsi0 - 0.5 * cb0)
            #print(type(cssh), cssh.function_space())

            cfs_dg = self.cfs_dg #FunctionSpace(cmesh, "DG", 1)
            cq0 = project(cq0, cfs_dg)
            cb0 = project(cb0, cfs_dg)

        if 1:
            cssh.rename("SSH")
            cq0.rename("PotentialVorticity")
            cb0.rename("Buoyancy")
            cpsi0.rename("Streamfunction")

            data_output_name = cpde_data_fname(self.file_work_space, file_index, self.output_sub_dir)

            if not output_file_visual == None:
                _t = round(time_stamp, 5)
                #_t = round(Dt * dumpfreq * file_index, 5)
                output_file_visual.write(cq0, cb0, cpsi0, cssh, time=_t)

            with DumbCheckpoint(data_output_name, single_file=True, mode=FILE_CREATE, comm=spatial_comm) as data_chk:
                data_chk.store(cq0)
                data_chk.store(cb0)
                data_chk.store(cpsi0)
                data_chk.store(cssh)

            #data_chk.close()

        #return np.asarray(kinetic_energy_series), np.asarray(potential_energy_series), np.asarray(total_energy_series), np.asarray(casimir_series), np.asarray(non_casimir_series)


    def coarse_grain_and_project(self, func_fine_res, field="Streamfunction"):
        cnx = self.cnx
        cny = self.cny

        k_sqr = cnx * cny #* 256 / cnx * 256 / cny

        flag_direct_subsample = False

        #Helmhotlz solver -- u is the solution
        u = None
        #fs = func_fine_res.function_space()
        cg_fs = self.Vcg  # fine res cg space 
        f = Function(cg_fs).project(func_fine_res)

        if flag_direct_subsample == False:
            u = TrialFunction(cg_fs)
            v = TestFunction(cg_fs)
            c_sqr = Constant(1.0 / k_sqr)

            a = (c_sqr * dot(grad(v), grad(u)) + v * u) * dx
            L = f * v * dx

            bc = DirichletBC(cg_fs, f, 'on_boundary') #if field == "Streamfunction" else []

            u = Function(cg_fs)
            solve(a == L, u, bcs=bc, solver_parameters={'ksp_type': 'cg'})
        else:
            u = f

        #project
        #cmesh = TorusMeshHierarchy(cnx, cny, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
        #cvfs = VectorFunctionSpace(cmesh, "CG", 1)
        #cfs = FunctionSpace(cmesh, "CG", 1)
        cvfs = self.cvfs
        cfs = self.cfs_cg
        ccoords = self.ccoords
        cfunc = Function(cfs)
        u_at_values = np.asarray(u.at(ccoords.dat.data, tolerance=1e-10))
        assert cfunc.dat.data.shape == u_at_values.shape 

        cfunc.dat.data[:] += u_at_values 

        return cfunc

    def observation_mesh(self, t, number_of_track_samples=100):
        """
        generates observation mesh that corresponds to two satellite tracks

        Input t is pde saved data file index
        Time interval between two consecutively numbered data files  are defined by dumpfreq
        in the solver
        Here we assume the dump_freq = 10        
        and dt = 0.0001 which is roughtly 3.5 minutes
        """
        r = 1./2./np.pi
        number_of_tracks = 4
        number_of_tracks2 = 5
        total_theta  = 2*np.pi * number_of_tracks
        total_theta2 = 2*np.pi * number_of_tracks2

        k_tilde = number_of_tracks/number_of_tracks2
        k_sum = number_of_tracks2 + number_of_tracks
        scaling = number_of_track_samples / k_sum

        ms = np.linspace(0, number_of_track_samples, number_of_track_samples, endpoint=False)
        ns = np.linspace(0, k_sum, k_sum, endpoint=False) 

        ns *= scaling
        ns = np.setdiff1d(ns, np.delete(ns, np.where(np.mod(ns,1)==0))).astype(int)
        ms_2 = np.setdiff1d(ms, ns)

        theta2 = total_theta2 / number_of_track_samples * ms_2
        theta = k_tilde*total_theta2 / number_of_track_samples * ms

        x2= theta2/total_theta2
        x = theta/total_theta #np.zeros(len(y))

        dump_freq = 10
        delta_t = 0.0001
        delta_t_per_save = dump_freq * delta_t  
        delta_t_per_day = 411 * delta_t
        delta_t_per_10_days = 4110 * delta_t


        c = t * 2 * np.pi * delta_t_per_save / delta_t_per_10_days
        
        # satellite 1
        z = r * np.sin(theta + c)
        y = r * np.cos(theta + c)

        # satellite 2
        z2 = r * np.sin(-theta2 + c)
        y2 = r * np.cos(-theta2 + c)

        #intersection points
        #z3 = r * np.sin(intersections * k_tilde + c)
        #y3 = r * np.cos(intersections * k_tilde + c)

        y  = np.arccos(z/r) *  np.sign(y) * r + 0.5
        y_ = np.arccos(z2/r) * np.sign(y2) * r + 0.5
        #y__= np.arccos(z3/r) * np.sign(y3) * r + 0.5

        # now we collect together the two satellite tracks 
        # without any duplicate points
        combine_xs = np.concatenate((x, x2))
        combine_ys = np.concatenate((y, y_))

        combined = np.vstack((combine_xs, combine_ys)).T
        #print(len(x), len(x2), scaling)

        return combined


if __name__ == "__main__":
    nx = 512
    cnx = 32 #128 #64
    T = 0.05
    dt = 0.0001
    dumpfreq = 10
    solve_flag = False
    write_visual = False 

    #batch_id = int(sys.argv[1])
    #print("batch_id: ", batch_id, ", ensemble_member_id: ",  ensemble_comm.rank + batch_id*ensemble_comm.size)

    ensemble_size = ensemble_comm.size 

    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=spatial_comm).get_fine_mesh()
    
    # It doesn't matter what the actual params are, we just need a param object to pass to the Solver constructor
    angry_dolphin_params = Example2params(T, dt, mesh, bc='x', alpha=None)
    
    wspace = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

    cg_data_generator = CoarseGrainedSolutionGenerator(angry_dolphin_params, ensemble, wspace, cnx, cnx)
    if 0: 
        visual_out = File(wspace.output_name("cpde_data.pvd", "cPDESolution/visuals"), comm=spatial_comm)
        #dat_range = np.arange((batch_id-1) * 375, batch_id*375) if batch_id < 4 else np.arange(3 * 375, 1501)
        dat_range =  np.arange(0, 1501)
        _t = 0
        for findex in dat_range:
            _t = np.round(dt*dumpfreq*findex, 5)
            # drumpfreq below doesn't do anything
            print(findex, _t, flush=True)
            cg_data_generator.generate_a_coarse_grained_data(ensemble, findex, dumpfreq, visual_out, time_stamp= _t )
            #_t += dt * dumpfreq

    if 0:
        from celluloid import Camera
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4,4))
        fig.tight_layout()
        
        ax2 = fig.add_subplot(1,1,1)
        ax2.set_aspect(aspect='equal')
        camera = Camera(fig)

        for findex in np.arange(0, 151):
            print(findex)
            values = cg_data_generator.generate_coarse_grained_data_at_obs_sites(ensemble, findex, dumpfreq)
            #print(values.shape)
            obs_sites = cg_data_generator.observation_mesh(findex, cg_data_generator.number_of_obs_per_satellite)
            #print(obs_sites.shape)
            #ax2.scatter(values[:,0], values[:,1], s=0.1, c='blue')
            ax2.scatter(obs_sites[:,0], obs_sites[:,1], c=values, cmap='turbo', vmin=-1, vmax=1, s=0.2)
            #ax2.scatter(xs[:, 0], xs[:,1], c=dat, cmap='turbo', vmin=-.5, vmax=0.5, s=0.2)
            camera.snap()

        animation = camera.animate(blit=False, interval=200)
        animation_name = wspace.output_name('test_satellite_track_{}.mov'.format(cnx), 'TestFiles')
        animation.save(animation_name, codec='png', dpi=200)

    if 0:
        data = {} 
        data2 = {}
        for findex in np.arange(0, 1501):
            print(findex, 'obs_data')
            data["obs_data_{}".format(findex)] = cg_data_generator.generate_coarse_grained_data_at_obs_sites(ensemble, findex, dumpfreq)
            data2["obs_data_{}".format(findex)] = cg_data_generator.generate_fine_res_data_at_obs_sites(ensemble, findex, dumpfreq)

        # actual evaluations of ssh -- no obs noise
        np.savez(wspace.output_name("obs_data", "cPDESolution"), **data)
        np.savez(wspace.output_name("obs_data_fs", "PDESolution"), **data2)

    if 1:
        ### save complete grid values for mse vs spread plot ###
        data = {}
        for findex in np.arange(0, 1501):
            data["obs_data_full_grid_{}".format(findex)] = cg_data_generator.generate_coarse_grained_data_at_all_grid_points(ensemble, findex, dumpfreq)
            print(findex, data["obs_data_full_grid_{}".format(findex)].shape)

        np.savez(wspace.output_name("obs_data_full_grid", "cPDESolution"), **data)

    ### generate sub observations at reduced satellite observation sites ####
    if 1:
        # load npz data that was stored in line 324
        all_obs = np.load(wspace.output_name('obs_data.npz', 'cPDESolution'))

        # load observation indices that were precomputed in accordance with the sample covariance estimator's dimension reduction
        sub_obs_site_indices = np.load(wspace.output_name('sub_obs_site_indices.npy', 'ParamFiles'))
        print(sub_obs_site_indices.shape)

        reduced_data = {}
        for _file in all_obs.files:
            #print(_file, '\n', all_obs[_file][sub_obs_site_indices][:3])
            #print(_file, '\n', all_obs[_file][sub_obs_site_indices].shape)
            reduced_data[_file] = all_obs[_file][sub_obs_site_indices]
            
        np.savez(wspace.output_name("obs_data_reduced", "cPDESolution"), **reduced_data)

