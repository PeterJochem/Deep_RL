// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2019 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
//This demo simulate a plate from into a box area of granular materials with certain attack and intrusion angles

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/core/ChGlobal.h"
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChForce.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/timestepper/ChTimestepper.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
#include "chrono_granular/physics/ChGranularTriMesh.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono_granular/utils/ChGranularJsonParser.h"

using namespace chrono;
using namespace chrono::granular;

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <json_file>" << std::endl;
}

void writeMeshFrames(std::ostringstream& outstream, ChBody& body, std::string obj_name, float mesh_scaling) {
    outstream << obj_name << ",";

    // Get frame position
    ChFrame<> body_frame = body.GetFrame_REF_to_abs();
    ChQuaternion<> rot = body_frame.GetRot();
    ChVector<> pos = body_frame.GetPos();

    // Get basis vectors
    ChVector<> vx = rot.GetXaxis();
    ChVector<> vy = rot.GetYaxis();
    ChVector<> vz = rot.GetZaxis();

    // Output in order
    outstream << pos.x() << ",";
    outstream << pos.y() << ",";
    outstream << pos.z() << ",";
    outstream << vx.x() << ",";
    outstream << vx.y() << ",";
    outstream << vx.z() << ",";
    outstream << vy.x() << ",";
    outstream << vy.y() << ",";
    outstream << vy.z() << ",";
    outstream << vz.x() << ",";
    outstream << vz.y() << ",";
    outstream << vz.z() << ",";
    outstream << mesh_scaling << "," << mesh_scaling << "," << mesh_scaling;
    outstream << "\n";
}
// define some constant for different stages of simulation
const double time_settle = 2.0;
const double particle_moving = 2.5;
const double plate_intrude = 1.0;
constexpr float F_CGS_TO_SI = 1e-5;
int main(int argc, char* argv[]) {

    std::ofstream data_set("data_sets/dset_fixed_depth.csv", std::ios_base::app);
    sim_param_holder params;
    if (argc < 2 || ParseJSON(argv[1], params) == false) {
        //ShowUsage(argv[0]);
        return 1;
    }

    // Get the command line arguments specifying gamma and the speed
    double gamma_launch = atof(argv[2]);
    double speed_launch = atof(argv[3]);
   	
    std::cout << gamma_launch << std::endl;
    std::cout << speed_launch << std::endl;

    float iteration_step = params.step_size;

    ChGranularChronoTriMeshAPI apiSMC_TriMesh(params.sphere_radius, params.sphere_density,
                                              make_float3(params.box_X, params.box_Y, params.box_Z));

    ChSystemGranularSMC_trimesh& gran_sys = apiSMC_TriMesh.getGranSystemSMC_TriMesh();
    double fill_bottom = -params.box_Z / 2.0; // -200/2
    double fill_top = params.box_Z / 2.0;     // 200/4

   // chrono::utils::PDSampler<float> sampler(2.4f * params.sphere_radius);
    chrono::utils::HCPSampler<float> sampler(2.1f * params.sphere_radius);

    // leave a 0.5cm margin at edges of sampling
    ChVector<> hdims(params.box_X / 2-0.5 , params.box_Y / 2-0.5 , 0);
    ChVector<> center(0, 0, fill_bottom + 2.0 * params.sphere_radius);
    std::vector<ChVector<float>> body_points;

    // Shift up for bottom of box
    std::vector<float3> particle_vel;
    center.z() += 3 * params.sphere_radius;
    while (center.z() < fill_top) {
        //std::cout << "Create layer at " << center.z() << std::endl;
        auto points = sampler.SampleBox(center, hdims);
        body_points.insert(body_points.end(), points.begin(), points.end());
        center.z() += 2.05 * params.sphere_radius;
    }

    apiSMC_TriMesh.setElemsPositions(body_points);
    //gran_sys.setParticlePositions(body_points);
    std::vector<float3> pointsFloat3;
    convertChVector2Float3Vec(body_points, pointsFloat3);
    int j = 0;
    float vvx, vvy, vvz;
    srand(time(NULL));
    for (j = 0; j < pointsFloat3.size(); j++) {
        vvx = pow(-1,rand()%2)*(rand() % 5+40);
        vvy = pow(-1,rand()%2)*(rand() % 5+40);
        vvz = pow(-1,rand()%2)*(rand() % 5+40);
        particle_vel.push_back(make_float3(vvx,vvy,vvz));
     //   std::cout << "vx= " << vvx << std::endl;
    }

   // std::cout <<"vector size:  "<< pointsFloat3.size()<<std::endl;
    gran_sys.setParticlePositions(pointsFloat3, particle_vel);
    gran_sys.set_BD_Fixed(true);
    std::function<double3(float)> pos_func_wave = [&params](float t) {
        double3 pos = {0, 0, 0};

        double t0 = 0.5;
        double freq = CH_C_PI / 4;

        if (t > t0) {
            pos.x = 0.1 * params.box_X * std::sin((t - t0) * freq);
        }
        return pos;
    };

    // gran_sys.setBDWallsMotionFunction(pos_func_wave);

    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);
    gran_sys.set_K_n_SPH2MESH(params.normalStiffS2M);

    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);
    gran_sys.set_Gamma_n_SPH2MESH(params.normalDampS2M);

    gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);
    gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);
    gran_sys.set_K_t_SPH2MESH(params.tangentStiffS2M);

    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
    gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
    gran_sys.set_Gamma_t_SPH2MESH(params.tangentDampS2M);

    gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    gran_sys.set_Adhesion_ratio_S2M(params.adhesion_ratio_s2m);
    gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

    gran_sys.set_fixed_stepSize(params.step_size);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
    gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
    gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);
    gran_sys.set_static_friction_coeff_SPH2MESH(params.static_friction_coeffS2M);

    std::string mesh_filename(GetChronoDataFile("../../foot.obj")); 
    std::string mesh_filename_lid = GetChronoDataFile("../../foot.obj");

   // std::vector<string> mesh_filenames(1,mesh_filename);

 //   std::vector<float3> mesh_translations(1, make_float3(-20.f, -20.f, 0.f));
    std::vector<string> mesh_filenames;
    std::vector<float3> mesh_translations;
    std::vector<float> mesh_masses;
    mesh_filenames.push_back(mesh_filename);
    mesh_filenames.push_back(mesh_filename_lid);

    float3 translation_plate = make_float3( 0.f, 0.f, 0.f);
    float3 translation_lid=make_float3(-50.f, -50.f, 0.f);
    mesh_translations.push_back(translation_plate);
    mesh_translations.push_back(translation_lid);


    float ball_radius = 20.f;
   	
    // Over what interval should we vary these?
    float length = 5.0;
    float width = 5.0;
    float thickness = 0.5;
    
    
    float lid_length = 100.0;
    float lid_width = 100.0;
    float lid_thickness = 10.0;

    std::vector<ChMatrix33<float>> mesh_rotscales(1, ChMatrix33<float>(0.5));
    mesh_rotscales.push_back(ChMatrix33<float>(2.0));


    float plate_density = 80.0;//params.sphere_density / 100.f;
    float plate_mass = (float)length * width * thickness * plate_density ;
    float lid_density = 0.001;
    float lid_mass = (float)lid_length * lid_width * lid_thickness * lid_density;


   // std::vector<float> mesh_masses(1, plate_mass);
   // std::vector<float> mesh_masses(1, lid_mass);
    mesh_masses.push_back(plate_mass);
    mesh_masses.push_back(lid_mass);
    
    apiSMC_TriMesh.load_meshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses);

    gran_sys.setOutputMode(params.write_mode);
    gran_sys.setVerbose(params.verbose);
    filesystem::create_directory(filesystem::path(params.output_dir));
    
    unsigned int nSoupFamilies = gran_sys.getNumTriangleFamilies();
    //std::cout << nSoupFamilies << " soup families" << std::endl;
    double* meshPosRot = new double[7 * nSoupFamilies];
    float* meshVel = new float[6 * nSoupFamilies]();

    gran_sys.initialize();

// create a plate for simulation
    ChSystemSMC sys_plate;
    sys_plate.SetContactForceModel(ChSystemSMC::ContactForceModel::Hooke);
    sys_plate.SetTimestepperType(ChTimestepper::Type::EULER_EXPLICIT);
    sys_plate.Set_G_acc(ChVector<>(0, 0, -980));
  //  auto rigid_plate = std::make_shared<ChBodyEasyBox>(length, width, thickness, plate_density, true, true);


    std::shared_ptr<ChBody> rigid_plate(sys_plate.NewBody());
    std::shared_ptr<ChBody> rigid_lid(sys_plate.NewBody());


    rigid_lid->SetMass(lid_mass);
    rigid_lid->SetPos(ChVector<>(0,0,15));
    rigid_plate->SetMass(plate_mass);
    rigid_plate->SetPos(ChVector<>(0,0,15));

   
    double inertiax = 1.0 / 12.0 * plate_mass*(thickness* thickness +width*width);
    double inertiay = 1.0 / 12.0 * plate_mass * (thickness * thickness + length * length);
    double inertiaz = 1.0 / 12.0 * plate_mass * (length * length + width * width);
    double lid_inertiax = 1.0 / 12.0 * lid_mass * (lid_thickness * lid_thickness + lid_width * lid_width);
    double lid_inertiay = 1.0 / 12.0 * lid_mass * (lid_thickness * lid_thickness + lid_length * lid_length);
    double lid_inertiaz = 1.0 / 12.0 * lid_mass * (lid_length * lid_length + lid_width * lid_width);

    rigid_lid->SetInertiaXX(ChVector<>(lid_inertiax,lid_inertiay,lid_inertiaz));
    rigid_lid->SetBodyFixed(true);
    rigid_plate->SetInertiaXX(ChVector<>(inertiax, inertiay, inertiaz));
    rigid_plate->SetBodyFixed(true);
    sys_plate.AddBody(rigid_plate); 
    sys_plate.AddBody(rigid_lid);

    //create a reference body that the plate can move vertically
    auto ref_body = chrono_types::make_shared<ChBody>();
    sys_plate.AddBody(ref_body);
    ref_body->SetBodyFixed(true);
    ref_body->SetCollide(false);
    ref_body->SetPos(ChVector<>(0, 0, 0));
    auto my_Link_reffoot = std::make_shared<ChLinkLockPrismatic>();
    my_Link_reffoot->Initialize(ref_body, rigid_plate, ChCoordsys<>(ChVector<>(0.0, 0.0,0.0), Q_from_AngZ(0)));
  //  sys_plate.AddLink(my_Link_reffoot);

    unsigned int out_fps = 50;
    //std::cout << "Rendering at " << out_fps << "FPS" << std::endl;

    unsigned int out_steps = (unsigned int)(1.0 / (out_fps * iteration_step));

    int currframe = 0;
    unsigned int curr_step = 0;

    gran_sys.disableMeshCollision();
    clock_t start = std::clock();
    //create an array to store a list of initial velocities
    //double vz_array[10] = { -1.0,-2.0,-5.0,-10.0,-20.0,-30.0,-40.0,-50.0,-100.0,-10000.0 };
    double beta_array[6] = {CH_C_PI/2.0, CH_C_PI/3.0, CH_C_PI/6.0, 0, -CH_C_PI/6.0, -CH_C_PI/3.0};
    double lid_speed = -5.0;
    bool change_gravity = false;
    bool paticle_moving_state = false;
    
    
    bool granular_set_state = false;
    gran_sys.disableMeshCollision();
    for(int i=0;i<6;i++)
    {   
        // the state of the plate, flase means the plate is fixed while true means the plate has been released
        bool plate_released = false;
        bool lid_released = false;
        bool plate_impact_state = false;
        bool plate_extract_state = false;
        double max_z = gran_sys.get_max_z();

        double gamma = CH_C_PI / 4.0;//CH_C_PI/2.0;
        double beta = beta_array[i];
        ///////////////////////////////////////////////////////////
        ////////// You can change the extract speed here//////////
        //////////////////////////////////////////////////////////

        double vx = -speed_launch * cos(gamma);
        double vz = -speed_launch * sin(gamma);
       	std::cout << "beta =  " << beta*180/CH_C_PI << " gamma =  " << gamma*180/CH_C_PI << std::endl;	

	//out_as << gamma*180/CH_C_PI << ',' << beta*180/CH_C_PI << std::endl; 
        //out_pos << gamma*180/CH_C_PI << ',' << beta*180/CH_C_PI << std::endl;
        //out_vel << gamma*180/CH_C_PI << ',' << beta*180/CH_C_PI << std::endl;

        double start_height = length / 2 * sin(beta);

        int counter = 0;
        rigid_plate->SetPos(ChVector<>(0, 0, 20));
        rigid_plate->SetBodyFixed(true);
        rigid_lid->SetPos(ChVector<>(0, 0, 200));
        rigid_lid->SetBodyFixed(true);
        for (double t = 0; t < (double)params.time_end; t += iteration_step, curr_step++) {
        
///////////////////////If statements for preparing the bed ////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////// 1. particle moving period: give particles initial velcoties and let it bounce                            ///////////////////////////////
////////////////////// 2. particles finish settling down & plate intrude: let the plate intrude into the granular domain for   ////////////////////////////////
/////////////////////     a certain depth and stop, and wait for 2s for granualr particles settling down                       ////////////////////////////////
////////////////////// 3. plate extract: make plate extract from the depth that it stopped                                     ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
           if (t >= particle_moving && plate_impact_state==false) {
                gran_sys.enableMeshCollision();
                max_z = gran_sys.get_max_z();
                rigid_plate->SetBodyFixed(false);
                // set the initial position of the plate 
                //rigid_plate->SetPos(ChVector<>(0, 0, max_z+ abs(start_height)));
                rigid_plate->SetPos(ChVector<>(0, 0, max_z + abs(start_height)));
		plate_impact_state = true;
              //  sys_plate.AddLink(my_Link_reffoot);
            }

           if (t >= particle_moving  && t<=particle_moving+plate_intrude && plate_impact_state == true) {

               rigid_plate->SetPos_dt(ChVector<>(0, 0, -10));
               rigid_plate->SetRot(Q_from_AngAxis(beta, VECT_Y));
           }

           if (t >= particle_moving + plate_intrude && t <= particle_moving + plate_intrude + time_settle) {
               rigid_plate->SetBodyFixed(true);
           }
           //  Before the plate starting extracting, it spends particle_moving + plate_intrude + time_settle = 5.5s for preparations,
           //  you should set a proper time_end in jason file to make you plate moves inside the granular box based on the extract speed you choose
           if (t >= particle_moving + plate_intrude + time_settle) {
               rigid_plate->SetBodyFixed(false);
               rigid_plate->SetPos_dt(ChVector<>(vx, 0, vz));
               rigid_plate->SetRot(Q_from_AngAxis(beta, VECT_Y));
           }
    
            auto plate_pos = rigid_plate->GetPos();
            auto plate_rot = rigid_plate->GetRot();
      
            auto plate_vel = rigid_plate->GetPos_dt();
        
            auto plate_ang_vel = rigid_plate->GetWvel_loc();
            plate_ang_vel = rigid_plate->GetRot().GetInverse().Rotate(plate_ang_vel);
            
            auto lid_pos = rigid_lid->GetPos();
            auto lid_rot = rigid_lid->GetRot();

            auto lid_vel = rigid_lid->GetPos_dt();
            auto lid_ang_vel = rigid_lid ->GetWvel_loc(); 

            lid_ang_vel = rigid_lid->GetRot().GetInverse().Rotate(lid_ang_vel);
            
            meshPosRot[0] =  plate_pos.x();
            meshPosRot[1] =  plate_pos.y();
            meshPosRot[2] = plate_pos.z();
            meshPosRot[3] =  plate_rot[0];
            meshPosRot[4] =  plate_rot[1];
            meshPosRot[5] =  plate_rot[2];
            meshPosRot[6] =  plate_rot[3];
            
            meshPosRot[0+7] = lid_pos.x();
            meshPosRot[1+7] = lid_pos.y();
            meshPosRot[2+7] = lid_pos.z();
            meshPosRot[3+7] = lid_rot[0];
            meshPosRot[4+7] = lid_rot[1];
            meshPosRot[5+7] = lid_rot[2];
            meshPosRot[6+7] = lid_rot[3];

            
            meshVel[0] = (float) plate_vel.x();
            meshVel[1] = (float) plate_vel.y();
            meshVel[2] = (float) plate_vel.z();
            meshVel[3] = (float) plate_ang_vel.x();
            meshVel[4] = (float) plate_ang_vel.y();
            meshVel[5] = (float) plate_ang_vel.z();
            
            meshVel[0+6] = (float)lid_vel.x();
            meshVel[1+6] = (float)lid_vel.y();
            meshVel[2+6] = (float)lid_vel.z();
            meshVel[3+6] = (float)lid_ang_vel.x();
            meshVel[4+6] = (float)lid_ang_vel.y();
            meshVel[5+6] = (float)lid_ang_vel.z();

            gran_sys.meshSoup_applyRigidBodyMotion(meshPosRot, meshVel);

            gran_sys.advance_simulation(iteration_step);
            sys_plate.DoStepDynamics(iteration_step);

            float plate_force[6*2];
            gran_sys.collectGeneralizedForcesOnMeshSoup(plate_force);

            rigid_plate->Empty_forces_accumulators();
            rigid_plate->Accumulate_force(ChVector<>(plate_force[0], plate_force[1], plate_force[2]), plate_pos, false);
            rigid_plate->Accumulate_torque(ChVector<>(plate_force[3], plate_force[4], plate_force[5]), false);
            rigid_lid->Empty_forces_accumulators();
            rigid_lid->Accumulate_force(ChVector<>(plate_force[6], plate_force[7], plate_force[8]), lid_pos, false);
            rigid_lid->Accumulate_torque(ChVector<>(plate_force[9], plate_force[10], plate_force[11]),false);


            // this part is for displaying some realtime data on command line and write down some position, velocity and force data into csv file
            if (counter % 20 == 0) {

		// Gamma, Beta, depth, position_x, position_z, velocity x, velocity y, GRF x, GRF Z	
		double depth = gran_sys.get_max_z() - plate_pos[2];
		
		double position_x = rigid_plate->GetPos()[0];
		double position_y = rigid_plate->GetPos()[1];
		double position_z = rigid_plate->GetPos()[2];
		
		double x_dt = rigid_plate->GetPos_dt()[0];
		double y_dt = rigid_plate->GetPos_dt()[1];
		double z_dt = rigid_plate->GetPos_dt()[2];

		double grf_x = plate_force[0] * F_CGS_TO_SI;
		double grf_y = plate_force[1] * F_CGS_TO_SI;
		double grf_z = plate_force[2] * F_CGS_TO_SI;

		// These are now in Nm
		double torque_x = 1e-7 * plate_force[3];
		double torque_y = 1e-7 * plate_force[4];
		double torque_z = 1e-7 * plate_force[5];
		
		data_set << t << ", " << gamma << ", " << beta << ", " << depth << ", " << position_x << ", " << position_y << ", " << position_z << ", " << x_dt << ", " << y_dt << ", " << z_dt << ", " << plate_ang_vel[0] << ", " << plate_ang_vel[1] << ", " << plate_ang_vel[2] << ", " << torque_x << ", " << torque_y << ", " << torque_z << ", " << grf_x << ", " << grf_y << ", " << grf_z << "\n";   
		
	}
	    counter++;
        }
     }
    clock_t end = std::clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cout << "Time: " << total_time << " seconds" << std::endl;

    delete[] meshPosRot;
    delete[] meshVel;
    data_set.close();
    return 0;
}
