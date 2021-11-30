#include <iostream>
#include <thread>
#include <atomic>
#include <ctime>
#include <csignal>
#include <exception>
#include "glfwManager.h"
#include "concurrency.h"
#include "MockD435iCamera.h"
#include "Util.h"
#include "OkvisSLAMSystem.h"

using namespace ark;
using boost::filesystem::path;

void signal_handler(int signum)
{
    cout << "Interrupt signal (" << signum << ") received.\n";
    exit(signum);
}


// https://learnopencv.com/rotation-matrix-to-euler-angles/
// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Eigen::Vector3d rotationMatrixToEulerAngles(Eigen::Matrix3d &R)
{
	float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

	bool singular = sy < 1e-6; // If

	double x, y, z;
	if (!singular)
	{
		x = atan2(R(2, 1), R(2, 2));
		y = atan2(-R(2, 0), sy);
		z = atan2(R(1, 0), R(0, 0));
	}
	else
	{
		x = atan2(-R(1, 2), R(1, 1));
		y = atan2(-R(2, 0), sy);
		z = 0;
	}
	return Eigen::Vector3d(x, y, z);

}

int main(int argc, char **argv)
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGILL, signal_handler);
    std::signal(SIGABRT, signal_handler);
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGTERM, signal_handler);
    if (argc > 5)
    {
        std::cerr << "Usage: ./" << argv[0] << " [configuration-yaml-file] [vocabulary-file] [skip-first-seconds] [data_path]" << std::endl
                  << "Args given: " << argc << std::endl;
        return -1;
    }

    google::InitGoogleLogging(argv[0]);

    // read configuration file
    std::string configFilename;
    if (argc > 1)
        configFilename = argv[1];
    else
        configFilename = util::resolveRootPath("config/d435i_intr.yaml");

    std::string vocabFilename;
    if (argc > 2)
        vocabFilename = argv[2];
    else
        vocabFilename = util::resolveRootPath("config/brisk_vocab.bn");

    okvis::Duration deltaT(0.0);
    if (argc > 3)
    {
        deltaT = okvis::Duration(atof(argv[3]));
    }

    path dataPath{"./data_path_25-10-2019 16-47-28"};
    if (argc > 4)
    {
        dataPath = path(argv[4]);
    }

    OkvisSLAMSystem slam(vocabFilename, configFilename);

    //setup display
    if (!MyGUI::Manager::init())
    {
        fprintf(stdout, "Failed to initialize GLFW\n");
        return -1;
    }

    printf("Camera initialization started...\n");
    fflush(stdout);
    MockD435iCamera camera(dataPath);

    printf("Camera-IMU initialization complete\n");
    fflush(stdout);

    //Window for displaying the path
    MyGUI::CameraWindow traj_win("Traj Viewer", 640 * 2, 480 * 2);
    MyGUI::ARCameraWindow ar_win("AR Viewer", 640 * 2.5, 480 * 2.5, GL_LUMINANCE, GL_UNSIGNED_BYTE, 6.16403320e+02, 6.16171021e+02, 3.18104584e+02, 2.33643127e+02, 0.01, 100);
    traj_win.set_pos(640 * 2.5, 100);
    ar_win.set_pos(0, 100);
    std::map<int, MyGUI::Path *> pathMap;
    MyGUI::Axis axis1("axis1", .1); // origin position axis
    MyGUI::Axis axis2("axis2", 1); // camera axis
    MyGUI::Grid grid1("grid1", 10, 1);
    traj_win.add_object(&axis1);
    traj_win.add_object(&axis2);
    traj_win.add_object(&grid1);
    ar_win.add_object(&axis1);
    std::vector<MyGUI::Object *> cubes;
	std::vector<MyGUI::Object *> errorCubes; // cubes created for measure error
    std::vector<Eigen::Matrix4d> T_K_cubes;
    std::vector<MapKeyFrame::Ptr> K_cubes;
	// vector for saving camera T_WC info for measuring error
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> T_vectors;

    const bool hideInactiveMaps = true;

    int lastMapIndex_path = 0;
    //Recieves output from SLAM system and displays to the screen
    FrameAvailableHandler handler([&](MultiCameraFrame::Ptr frame) {
        Eigen::Affine3d transform(frame->T_WC(3));
        const auto mapIndex = slam.getActiveMapIndex();

        if (pathMap.find(mapIndex) == pathMap.end()) {
            string name = "path"+std::to_string(mapIndex);
            pathMap[mapIndex] = new MyGUI::Path{name, Eigen::Vector3d(1, 0, 0)};
            traj_win.add_object(pathMap[mapIndex]);
        }
        if (lastMapIndex_path != mapIndex) {
            if (hideInactiveMaps) {
                pathMap[lastMapIndex_path]->clear();
            }
            lastMapIndex_path = mapIndex;
        }
        pathMap[mapIndex]->add_node(transform.translation());
        axis2.set_transform(transform);
		//Eigen::Affine3d pose = axis2.get_pose();
		//std::cout << "handler: axis2.pos: " << pose.matrix() << std::endl;
        ar_win.set_camera(transform);
        ar_win.set_image(frame->images_[3]);
        if (ar_win.clicked())
        {
            std::string cube_name = std::string("CubeNum") + std::to_string(cubes.size());
            MyGUI::Object *obj = new MyGUI::Cube(cube_name, 0.1, 0.1, 0.1);
            obj->set_transform(transform);
            cubes.push_back(obj);
            T_K_cubes.push_back(frame->T_KS_);
            K_cubes.push_back(frame->keyframe_);
            std::cout << "Adding cube " << cube_name << std::endl;
            ar_win.add_object(obj); //NOTE: this is bad, should change objects to shared_ptr
			//frame->saveSimple("map_images/");
			//T_vectors.push_back(frame->getTransformMatrix());
			//std::cout << "Saving transformation matrix..." << std::endl;
        }
		if (ar_win.rPressed()) 
		{
			std::string cube_name = std::string("MarkerCube") + std::to_string(errorCubes.size());
			MyGUI::Object *obj = new MyGUI::Cube(cube_name, 0.05, 0.05, 0.05);
			obj->set_transform(transform);
			errorCubes.push_back(obj);
			T_K_cubes.push_back(frame->T_KS_);
			K_cubes.push_back(frame->keyframe_);
			std::cout << "Adding cube " << cube_name << std::endl;
			ar_win.add_object(obj); //NOTE: this is bad, should change objects to shared_ptr
			frame->saveSimple("map_images/");
			T_vectors.push_back(frame->getTransformMatrix());
			std::cout << "Saving transformation matrix..." << std::endl;
		}
    });
    slam.AddFrameAvailableHandler(handler, "mapping");

    KeyFrameAvailableHandler kfHandler([](MultiCameraFrame::Ptr frame) {
        frame->saveSimple("map_images/");
    });
    //slam.AddKeyFrameAvailableHandler(kfHandler, "saving");

    LoopClosureDetectedHandler loopHandler([&](void) {
        std::vector<Eigen::Matrix4d> traj;
        slam.getTrajectory(traj);
        const auto mapIndex = slam.getActiveMapIndex();
        pathMap[mapIndex]->clear();
        for (size_t i = 0; i < traj.size(); i++)
        {
            pathMap[mapIndex]->add_node(traj[i].block<3, 1>(0, 3));
        }
        for (size_t i = 0; i < cubes.size(); i++)
        {
            if (K_cubes[i] != nullptr)
                cubes[i]->set_transform(Eigen::Affine3d(K_cubes[i]->T_WS() * T_K_cubes[i]));
        }
    });
    slam.AddLoopClosureDetectedHandler(loopHandler, "trajectoryUpdate");

    SparseMapMergeHandler mergeHandler([&](int deletedIndex, int currentIndex) {
        pathMap[deletedIndex]->clear();
        std::vector<Eigen::Matrix4d> traj;
        slam.getMap(currentIndex)->getTrajectory(traj);
        pathMap[currentIndex]->clear();
        for (size_t i = 0; i < traj.size(); i++) {
            pathMap[currentIndex]->add_node(traj[i].block<3, 1>(0, 3));
        }
    });
    slam.AddSparseMapMergeHandler(mergeHandler, "mergeUpdate");

    //run until display is closed
    okvis::Time start(0.0);
    camera.start();
    int lastMapIndex = -1;

    while (MyGUI::Manager::running())
    {
        //Update the display
        MyGUI::Manager::update();

        try
        {
            //Get current camera frame
            MultiCameraFrame::Ptr frame(new MultiCameraFrame);
            camera.update(*frame);

            const auto frameId = frame->frameId_;
            if (frameId < 0) {
                std::cout << "Data end reached\n";
                break;
            }
            const auto &infrared = frame->images_[0];
            const auto &infrared2 = frame->images_[1];
            const auto &rgb = frame->images_[3];
            const auto &depth = frame->images_[4];

            cv::imshow(std::string(camera.getModelName()) + " Infrared", infrared);
            cv::imshow(std::string(camera.getModelName()) + " Depth", depth);

            std::vector<ImuPair> imuData;
            camera.getImuToTime(frame->timestamp_, imuData);

            //Add data to SLAM system
            slam.PushIMU(imuData);
            // make it the same as real camera
            frame->images_.resize(4);
            slam.PushFrame(frame);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n'; 
        }
        catch (...)
        {
            std::cout << "ex catched\n";
        }
        const auto isReset = slam.okvis_estimator_->isReset();
        const auto mapIndex = slam.getActiveMapIndex();
        if (mapIndex != lastMapIndex) {
            lastMapIndex = mapIndex;
            std::cout << "Mapnumber : " << mapIndex << "\n";
        }

        if (isReset) {
            traj_win.msg_ = " *Reseting*";
        } else {
            traj_win.msg_ = " ";
        }
        int k = cv::waitKey(1);
        if (k == 'q' || k == 'Q' || k == 27)
            break; // 27 is ESC
    }
	Eigen::Matrix4d startPose = T_vectors[0];
	Eigen::Matrix4d endPose = T_vectors[T_vectors.size() - 1];
	//double rotation_err = \
	//	(startPose(0, 0) - endPose(0, 0)) * (startPose(0, 0) - endPose(0, 0)) + (startPose(0, 1) - endPose(0, 1)) * (startPose(0, 1) - endPose(0, 1)) + (startPose(0, 2) - endPose(0, 2)) * (startPose(0, 2) - endPose(0, 2)) +
	//	(startPose(1, 0) - endPose(1, 0)) * (startPose(1, 0) - endPose(1, 0)) + (startPose(1, 1) - endPose(1, 1)) * (startPose(1, 1) - endPose(1, 1)) + (startPose(1, 2) - endPose(1, 2)) * (startPose(1, 2) - endPose(1, 2)) + 
	//	(startPose(2, 0) - endPose(2, 0)) * (startPose(2, 0) - endPose(2, 0)) + (startPose(2, 1) - endPose(2, 1)) * (startPose(2, 1) - endPose(2, 1)) + (startPose(2, 2) - endPose(2, 2)) * (startPose(2, 2) - endPose(2, 2));
	//rotation_err = sqrt(rotation_err);
	//double translation_err = (startPose(0, 3) - endPose(0, 3)) * (startPose(0, 3) - endPose(0, 3)) + (startPose(1, 3) - endPose(1, 3)) * (startPose(1, 3) - endPose(1, 3)) + (startPose(2, 3) - endPose(2, 3)) * (startPose(2, 3) - endPose(2, 3));
	//translation_err = sqrt(translation_err);
	//std::cout << "The rotation error is " << rotation_err << std::endl;
	//std::cout << "The translation error is " << translation_err << std::endl;
	//std::cout << "-------------------------------------" << std::endl;
	Eigen::Matrix3d R1 = startPose.block<3, 3>(0, 0);
	Eigen::Matrix3d R2 = endPose.block<3, 3>(0, 0);
	R2.transposeInPlace();
	Eigen::Matrix3d residualR = R1 * R2;
	Eigen::Vector3d eulerAngleR = rotationMatrixToEulerAngles(residualR);
	Eigen::Matrix4d diffPose = endPose - startPose;
	double rotation_err_eigen = residualR.norm();
	double translation_err_eigen = diffPose.col(3).norm();
	std::cout << "------------------------------------- Evaluation -------------------------------" << std::endl;
	std::cout << "The euler angle (degree) rotation error is: x: " << eulerAngleR.x() * (180 / M_PI) << "; y: " << eulerAngleR.y() * (180 / M_PI) << "; z: " << eulerAngleR.z() * (180 / M_PI) << std::endl;
	std::cout << "The rotation residual matrix is \n" << residualR << std::endl;
	// std::cout << "The rotation residual frobenius norm is " << rotation_err_eigen << std::endl;
	std::cout << "The eigen translation error is " << translation_err_eigen << std::endl;
    printf("\nTerminate...\n");
    // Clean up
    slam.ShutDown();
    printf("\nExiting...\n");
    return 0;
}