#include "optimizer.h"
#include "transform_util.h"
#include <fstream>
#include <random>

Mat Optimizer::eigen2mat(Eigen::MatrixXd A)
{
  Mat B;
  eigen2cv(A, B);

  return B;
}

Mat Optimizer::gray_gamma(Mat img)
{
  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY);
  double contrast = 1.1;
  double brightness = 0;
  double delta = 30;
  for (int i = 0; i < img.rows; i++)
  {
    for (int j = 0; j < img.cols; j++)
    {
      int g = gray.at<uchar>(i, j);
      gray.at<uchar>(i, j) = saturate_cast<uchar>(
          contrast * (gray.at<uchar>(i, j) - delta) + brightness);
    }
  }
  return gray;
}

double Optimizer::getPixelValue(Mat *image, float x, float y)
{
  // 法1：双线性插值
  uchar *data = &image->data[int(y) * image->step + int(x)]; // 以字节为单位计算图像起始位置到目标位置的偏移量
  float xx = x - floor(x);
  float yy = y - floor(y);
  return float((1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] +
               (1 - xx) * yy * data[image->step] +
               xx * yy * data[image->step + 1]);
}

vector<Point> Optimizer::readfromcsv(string filename)
{
  vector<Point> pixels;
  ifstream inFile(filename, ios::in);
  string lineStr;
  while (getline(inFile, lineStr))
  {
    Point pixel;
    istringstream record(lineStr);
    string x, y;
    record >> x;
    pixel.x = atoi(x.c_str());
    record >> y;
    pixel.y = atoi(y.c_str());
    pixels.push_back(pixel);
  }
  return pixels;
}

Mat Optimizer::tail(Mat img, string index)
{
  if (index == "f")
  {
    Rect m_select_f = Rect(0, 0, img.cols, sizef);
    Mat cropped_image_f = img(m_select_f);
    Mat border(img.rows - sizef, img.cols, cropped_image_f.type(),
               Scalar(0, 0, 0));
    Mat dst_front;
    vconcat(cropped_image_f, border, dst_front);
    return dst_front;
  }
  else if (index == "l")
  {
    Rect m_select_l = Rect(0, 0, sizel, img.rows);
    Mat cropped_image_l = img(m_select_l);
    Mat border2(img.rows, img.cols - sizel, cropped_image_l.type(),
                Scalar(0, 0, 0));
    Mat dst_left;
    hconcat(cropped_image_l, border2, dst_left);
    return dst_left;
  }
  else if (index == "b")
  {
    Rect m_select_b = Rect(0, img.rows - sizeb, img.cols, sizeb);
    Mat cropped_image_b = img(m_select_b);
    Mat border1(img.rows - sizeb, img.cols, cropped_image_b.type(),
                Scalar(0, 0, 0));
    Mat dst_behind;
    vconcat(border1, cropped_image_b, dst_behind);
    return dst_behind;
  }
  else if (index == "r")
  {
    Rect m_select_r = Rect(img.cols - sizer, 0, sizer, img.rows);
    Mat cropped_image_r = img(m_select_r);
    Mat border3(img.rows, img.cols - sizer, cropped_image_r.type(),
                Scalar(0, 0, 0));
    Mat dst_right;
    hconcat(border3, cropped_image_r, dst_right);
    return dst_right;
  }
  return Mat(img.rows, img.cols, img.type());
}

void Optimizer::SaveOptResult(const string filename)
{
  Mat opt_after = generate_surround_view(imgf_bev_rgb, imgl_bev_rgb,
                                         imgb_bev_rgb, imgr_bev_rgb);
  imwrite(filename, opt_after);
}

void Optimizer::show(string idx, string filename)
{
  Mat dst, dst1;

  if (idx == "right")
  { // first
    if (fixed == "front")
      addWeighted(imgf_bev_rgb, 0.5, imgr_bev_rgb, 0.5, 3, dst);
    else
      addWeighted(imgb_bev_rgb, 0.5, imgr_bev_rgb, 0.5, 3, dst);
    imwrite(filename, dst);
  }
  if (idx == "behind")
  { // second
    addWeighted(imgb_bev_rgb, 0.5, imgr_bev_rgb, 0.5, 3, dst);
    addWeighted(dst, 1, imgl_bev_rgb, 0.5, 3, dst1);
    imwrite(filename, dst1);
  }
  if (idx == "left")
  { // third
    if (fixed == "front")
      addWeighted(imgl_bev_rgb, 0.5, imgf_bev_rgb, 0.5, 3, dst);
    else
      addWeighted(imgl_bev_rgb, 0.5, imgb_bev_rgb, 0.5, 3, dst);
    imwrite(filename, dst);
  }
  if (idx == "front")
  {
    addWeighted(imgf_bev_rgb, 0.5, imgr_bev_rgb, 0.5, 3, dst);
    addWeighted(dst, 1, imgl_bev_rgb, 0.5, 3, dst1);
    imwrite(filename, dst1);
  }
}

void Optimizer::world2cam(double point2D[2], double point3D[3],
                          Eigen::Matrix3d K, vector<double> D)
{
  double norm = sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1]);
  double theta = atan(point3D[2] / norm);
  double t, t_i;
  double rho, x, y;
  double invnorm;
  int i;

  if (norm != 0)
  {
    invnorm = 1 / norm;
    t = theta;
    rho = D[0];
    t_i = 1;

    for (i = 1; i < D.size(); i++)
    {
      t_i *= t;
      rho += t_i * D[i];
    }

    x = point3D[0] * invnorm * rho;
    y = point3D[1] * invnorm * rho;

    point2D[0] = x * K(0, 0) + y * K(0, 1) + K(0, 2);
    point2D[1] = x * K(1, 0) + y + K(1, 2);
  }
  else
  {
    point2D[0] = K(0, 2);
    point2D[1] = K(1, 2);
  }
}

void Optimizer::distortPointsOcam(Mat &P_GC1, Mat &p_GC, Eigen::Matrix3d &K_C,
                                  vector<double> &D_C)
{
  double M[3];
  double m[2];
  for (int i = 0; i < P_GC1.cols; i++)
  {
    M[0] = P_GC1.at<Vec2d>(0, i)[0];
    M[1] = P_GC1.at<Vec2d>(0, i)[1];
    M[2] = -1;
    world2cam(m, M, K_C, D_C);
    p_GC.at<Vec2d>(0, i)[0] = m[0];
    p_GC.at<Vec2d>(0, i)[1] = m[1];
  }
}

void Optimizer::distortPoints(Mat &P_GC1, Mat &p_GC, Eigen::Matrix3d &K_C)
{
  for (int i = 0; i < P_GC1.cols; i++)
  {
    double x = P_GC1.at<Vec2d>(0, i)[0];
    double y = P_GC1.at<Vec2d>(0, i)[1];

    double u = x * K_C(0, 0) + K_C(0, 2);
    double v = y * K_C(1, 1) + K_C(1, 2);

    p_GC.at<Vec2d>(0, i)[0] = u;
    p_GC.at<Vec2d>(0, i)[1] = v;
  }
}

void Optimizer::initializeK()
{
  Eigen::Matrix3d K_F;
  Eigen::Matrix3d K_L;
  Eigen::Matrix3d K_B;
  Eigen::Matrix3d K_R;

  // common
  if (data_index == "common")
  {
    K_F << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750, 0.00000000,
        0.00000000, 1.00000000;
    // K_F << 452.277, 0, 323.62, 0, 452.277, 244.333, 0, 0, 1;
    K_L << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750, 0.00000000,
        0.00000000, 1.00000000;
    // K_L << 452.277, 0, 323.62, 0, 452.277, 244.333, 0, 0, 1;
    K_B << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750, 0.00000000,
        0.00000000, 1.00000000;
    K_R << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750, 0.00000000,
        0.00000000, 1.00000000;
  }

  // fisheye
  if (data_index == "fisheye")
  {
    K_F << 422.13163849, 0.00000000, 612.82890504, 0.00000000, 421.10340889,
        545.05656249, 0.00000000, 0.00000000, 1.00000000;
    K_L << 420.60079305, 0.00000000, 650.54173853, 0.00000000, 418.94827188,
        527.27178143, 0.00000000, 0.00000000, 1.00000000;
    K_B << 422.61569350, 0.00000000, 632.46019501, 0.00000000, 421.27373079,
        548.34673288, 0.00000000, 0.00000000, 1.00000000;
    K_R << 421.64203585, 0.00000000, 640.09362064, 0.00000000, 420.26647020,
        529.05566315, 0.00000000, 0.00000000, 1.00000000;
  }

  intrinsic_front = K_F;
  intrinsic_left = K_L;
  intrinsic_behind = K_B;
  intrinsic_right = K_R;
  return;
}

void Optimizer::initializeD()
{
  vector<double> D_F;
  vector<double> D_L;
  vector<double> D_B;
  vector<double> D_R;

  // common-pinhole
  if (data_index == "common")
  {
    D_F = {0, 0, 0, 0};
    D_L = {0, 0, 0, 0};
    D_B = {0, 0, 0, 0};
    D_R = {0, 0, 0, 0};
  }

  // fisheye
  if (data_index == "fisheye")
  {
    D_F = {-0.07031853, 0.00387505, -0.00333139, 0.00056406};
    D_L = {-6.58382798e-02, -2.00728513e-03, -3.72535694e-04, 1.81851668e-06};
    D_B = {-0.06553861, -0.00094857, -0.00150748, 0.000325};
    D_R = {-0.07289752, 0.01254629, -0.01300477, 0.00361266};
  }

  distortion_params_front = D_F;
  distortion_params_left = D_L;
  distortion_params_behind = D_B;
  distortion_params_right = D_R;
}
void Optimizer::initializePose()
{
  Mat rvecs_F, rvecs_L, rvecs_B, rvecs_R;
  Mat rotation_F, rotation_L, rotation_B, rotation_R;
  Mat translation_F, translation_L, translation_B, translation_R;
  Mat T_FG, T_LG, T_BG, T_RG;
  vector<double> D_F;
  vector<double> D_L;
  vector<double> D_B;
  vector<double> D_R;
  Mat K1_F;
  eigen2cv(intrinsic_front, K1_F);
  Mat K1_L;
  eigen2cv(intrinsic_left, K1_L);
  Mat K1_R;
  eigen2cv(intrinsic_right, K1_R);
  Mat K1_B;
  eigen2cv(intrinsic_behind, K1_B);
  Mat v;
  v = (Mat_<double>(1, 4) << 0, 0, 0, 1);
  solvePnP(pts_3d_F, pts_2d_F, K1_F, distortion_params_front, rvecs_F, translation_F);
  Rodrigues(rvecs_F, rotation_F);
  hconcat(rotation_F, translation_F, T_FG);
  T_FG.push_back(v);
  cout << "T_FG" << T_FG << endl;

  solvePnP(pts_3d_L, pts_2d_L, K1_L, distortion_params_left, rvecs_L, translation_L);
  Rodrigues(rvecs_L, rotation_L);
  hconcat(rotation_L, translation_L, T_LG);
  T_LG.push_back(v);
  cout << "T_LG" << T_LG << endl;

  solvePnP(pts_3d_R, pts_2d_R, K1_R, distortion_params_right, rvecs_R, translation_R);
  Rodrigues(rvecs_R, rotation_R);
  hconcat(rotation_R, translation_R, T_RG);
  T_RG.push_back(v);
  cout << "T_RG" << T_RG << endl;

  solvePnP(pts_3d_B, pts_2d_B, K1_B, distortion_params_behind, rvecs_B, translation_B);
  Rodrigues(rvecs_B, rotation_B);
  hconcat(rotation_B, translation_B, T_BG);
  T_BG.push_back(v);
  cout << "T_BG" << T_BG << endl;

  Eigen::Matrix4d T_FG1;
  Eigen::Matrix4d T_LG1;
  Eigen::Matrix4d T_BG1;
  Eigen::Matrix4d T_RG1;
  cv2eigen(T_FG, T_FG1);
  cv2eigen(T_LG, T_LG1);
  cv2eigen(T_RG, T_RG1);
  cv2eigen(T_BG, T_BG1);
  Eigen::Matrix4d T_CG;
  T_CG << 0, -1, 0, 0,
      -1, 0, 0, 0,
      0, 0, -1, 1.5,
      0, 0, 0, 1;
  extrinsic_front = T_FG1 * T_CG;
  cout << "extrinsic_front" << extrinsic_front << endl;
  extrinsic_left = T_LG1 * T_CG;
  extrinsic_behind = T_BG1 * T_CG;
  extrinsic_right = T_RG1 * T_CG;

  cout
      << "extrinsic_front:" << endl
      << extrinsic_front << endl;
  cout << "eular:" << endl
       << TransformUtil::Rotation2Eul(extrinsic_front.block(0, 0, 3, 3))
       << endl;
  cout << "extrinsic_left:" << endl
       << extrinsic_left << endl;
  cout << "eular:" << endl
       << TransformUtil::Rotation2Eul(extrinsic_left.block(0, 0, 3, 3)) << endl;
  cout << "extrinsic_right:" << endl
       << extrinsic_right << endl;
  cout << "eular:" << endl
       << TransformUtil::Rotation2Eul(extrinsic_right.block(0, 0, 3, 3))
       << endl;
  cout << "extrinsic_behind:" << endl
       << extrinsic_behind << endl;
  cout << "eular:" << endl
       << TransformUtil::Rotation2Eul(extrinsic_behind.block(0, 0, 3, 3))
       << endl;
  return;
}
// void Optimizer::initializePose()
// { // ground->camera
//   Eigen::Matrix4d T_FG;
//   Eigen::Matrix4d T_LG;
//   Eigen::Matrix4d T_BG;
//   Eigen::Matrix4d T_RG;

//   // common//应该是相机外参的GT
//   if (data_index == "common")
//   {
//     T_FG << 1, 0, 0, 0, 0, 0, 1, -4.1, 0, -1, 0, -2.5, 0, 0, 0, 1;
//     T_LG << 0, -1, 0, 0, 0, 0, 1, -4.1, -1, 0, 0, -1, 0, 0, 0, 1;
//     T_BG << -1, 0, 0, 0, 0, 0, 1, -4.1, 0, 1, 0, -2, 0, 0, 0, 1;
//     T_RG << 0, 1, 0, 0, 0, 0, 1, -4.1, 1, 0, 0, -1, 0, 0, 0, 1;
//   }

//   // ROCES
//   if (data_index == "fisheye")
//   {
//     T_FG << 9.99277118e-01, 3.82390286e-04, -3.80143958e-02, 6.75437418e-01,
//         -2.30748265e-02, -7.88582447e-01, -6.14495953e-01, 2.50896883e+01,
//         -3.02124625e-02, 6.14928921e-01, -7.88003572e-01, 3.17779305e+00, 0, 0,
//         0, 1;
//     T_LG << -1.21898860e-02, 9.99924056e-01, -1.81349393e-03, 1.36392943e+00,
//         8.02363600e-01, 8.69913885e-03, -5.96772133e-01, 1.60942881e+01,
//         -5.96711036e-01, -8.72966581e-03, -8.02408707e-01, 1.04105913e+01, 0, 0,
//         0, 1;
//     T_BG << -9.99615699e-01, 1.56439861e-02, -2.28849354e-02, 1.09266953e+00,
//         2.59906371e-02, 8.16008735e-01, -5.77454960e-01, 2.46308124e+01,
//         9.64060983e-03, -5.77827838e-01, -8.16101739e-01, 6.60957845e+00, 0, 0,
//         0, 1;
//     T_RG << 4.57647596e-03, -9.99989102e-01, 9.22798184e-04, -1.66115120e-01,
//         -6.26343448e-01, -3.58584197e-03, -7.79538984e-01, 1.76226207e+01,
//         7.79533797e-01, 2.98955282e-03, -6.26353033e-01, 6.08338205e+00, 0, 0,
//         0, 1;
//   }
//   // 给GT乘一个偏移量生成错误的外参
//   Eigen::Matrix4d left_disturbance;
//   Eigen::Matrix3d left_disturbance_rot_mat;
//   Vec3f left_disturbance_rot_euler; // R(euler)
//   Mat_<double> left_disturbance_t =
//       (Mat_<double>(3, 1) << 0.0095, 0.0025, -0.0086);
//   left_disturbance_rot_euler << 0.95, 1.25, -2.86;
//   left_disturbance_rot_mat =
//       TransformUtil::eulerAnglesToRotationMatrix(left_disturbance_rot_euler);
//   left_disturbance = TransformUtil::R_T2RT(
//       TransformUtil::eigen2mat(left_disturbance_rot_mat), left_disturbance_t);
//   T_LG *= left_disturbance;

//   Eigen::Matrix4d right_disturbance;
//   Eigen::Matrix3d right_disturbance_rot_mat;
//   Vec3f right_disturbance_rot_euler;
//   Mat_<double> right_disturbance_t =
//       (Mat_<double>(3, 1) << 0.0065, -0.0075, 0.0095);
//   right_disturbance_rot_euler << -2.95, 1.25, -2.8;
//   right_disturbance_rot_mat =
//       TransformUtil::eulerAnglesToRotationMatrix(right_disturbance_rot_euler);
//   right_disturbance = TransformUtil::R_T2RT(
//       TransformUtil::eigen2mat(right_disturbance_rot_mat), right_disturbance_t);
//   T_RG *= right_disturbance;

//   Eigen::Matrix4d behind_disturbance;
//   Eigen::Matrix3d behind_disturbance_rot_mat;
//   Vec3f behind_disturbance_rot_euler;
//   Mat_<double> behind_disturbance_t =
//       (Mat_<double>(3, 1) << -0.002, -0.0076, 0.0096);
//   behind_disturbance_rot_euler << -1.75, 2.95, -1.8;
//   behind_disturbance_rot_mat =
//       TransformUtil::eulerAnglesToRotationMatrix(behind_disturbance_rot_euler);
//   behind_disturbance = TransformUtil::R_T2RT(
//       TransformUtil::eigen2mat(behind_disturbance_rot_mat),
//       behind_disturbance_t);
//   T_BG *= behind_disturbance;
//   // 优化的就是这个错误的外参
//   extrinsic_front = T_FG;
//   extrinsic_left = T_LG;
//   // extrinsic_left << -0.6013330580895527, -0.7482116967800981, -0.2803173381189848, -0.1393619144902135,
//   //     0.791973905085537, -0.6045791467896465, -0.08521378368942238, 0.2131292634479425,
//   //     -0.1057160674270058, -0.2732458820706172, 0.9561175665258009, -0.9670336984054693,
//   //     0, 0, 0, 1;
//   // extrinsic_front << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

//   extrinsic_behind = T_BG;
//   extrinsic_right = T_RG;

//   cout << "extrinsic_front:" << endl
//        << extrinsic_front << endl;
//   cout << "eular:" << endl
//        << TransformUtil::Rotation2Eul(extrinsic_front.block(0, 0, 3, 3))
//        << endl;
//   cout << "extrinsic_left:" << endl
//        << extrinsic_left << endl;
//   cout << "eular:" << endl
//        << TransformUtil::Rotation2Eul(extrinsic_left.block(0, 0, 3, 3)) << endl;
//   cout << "extrinsic_right:" << endl
//        << extrinsic_right << endl;
//   cout << "eular:" << endl
//        << TransformUtil::Rotation2Eul(extrinsic_right.block(0, 0, 3, 3))
//        << endl;
//   cout << "extrinsic_behind:" << endl
//        << extrinsic_behind << endl;
//   cout << "eular:" << endl
//        << TransformUtil::Rotation2Eul(extrinsic_behind.block(0, 0, 3, 3))
//        << endl;
//   return;
// }

void Optimizer::initializeKG()
{
  // Eigen::Matrix3d K_G = Eigen::Matrix3d::Zero();

  // common
  if (data_index == "common")
  {
    int rows = 1500; // 图像的尺寸
    int cols = 1500;
    double dX = 0.005; // 虚拟相机的分辨率要求
    double dY = 0.005;
    cv::Mat K_G = cv::Mat::zeros(3, 3, CV_64FC1); // 设置bev虚拟相机的内参矩阵
    K_G.at<double>(0, 0) = 1 / dX;                // fx
    K_G.at<double>(1, 1) = 1 / dY;                // fy
    K_G.at<double>(0, 2) = cols / 2;              // cx
    K_G.at<double>(1, 2) = rows / 2;              // cy
    K_G.at<double>(2, 2) = 1.0;
    // K_G << 390.425287, 0.00000000, 750, 0.00000000, 390.425287, 750, 0.00000000,
    //     0.00000000, 1.00000000;
    // K_G << 452.277, 0, 323.62, 0, 452.277, 244.333, 0, 0, 1;
    cv2eigen(K_G, KG);
    // KG = K_G;
  }

  // fisheye
  if (data_index == "fisheye")
  {
    Eigen::Matrix3d K_G = Eigen::Matrix3d::Zero();
    K_G(0, 0) = 1 / 0.15;
    K_G(1, 1) = -1 / 0.15;
    K_G(0, 2) = bcols / 2;
    K_G(1, 2) = brows / 2;
    K_G(2, 2) = 1.0;
    KG = K_G;
  }
}

void Optimizer::initializeHeight()
{
  if (data_index == "common")
  { // common
    hf = 2.1;
    hl = 2.1;
    hb = 2.1;
    hr = 2.1;
  }
  else
  { // fisheye
    hf = 1;
    hl = 1;
    hb = 1;
    hr = 1;
  }
}

void Optimizer::initializetailsize()
{
  // common
  if (data_index == "common")
  {
    sizef = 450;
    sizel = 450;
    sizeb = 350;
    sizer = 450;
  }
  // fisheye
  if (data_index == "fisheye")
  {
    sizef = 340;
    sizel = 390;
    sizeb = 380;
    sizer = 390;
  }
}

// 角点相对于车体坐标系在3d空间中的位置
void Optimizer::Initialize3dPoints()
{

  pts_3d_F.push_back(Point3f(3.5, 0.12, 0.36));
  pts_3d_F.push_back(Point3f(3.5, -0.12, 0.36));
  pts_3d_F.push_back(Point3f(3.5, 0.36, 0.12));
  pts_3d_F.push_back(Point3f(3.5, 0.12, 0.12));
  pts_3d_F.push_back(Point3f(3.5, -0.12, 0.12));
  pts_3d_F.push_back(Point3f(3.5, -0.36, 0.12));
  pts_3d_F.push_back(Point3f(3.5, 0.36, -0.12));
  pts_3d_F.push_back(Point3f(3.5, 0.12, -0.12));
  pts_3d_F.push_back(Point3f(3.5, -0.12, -0.12));
  pts_3d_F.push_back(Point3f(3.5, -0.36, -0.12));
  pts_3d_F.push_back(Point3f(3.5, 0.12, -0.36));
  pts_3d_F.push_back(Point3f(3.5, -0.12, -0.36));

  pts_3d_L.push_back(Point3f(-0.12, 2.0, 0.36));
  pts_3d_L.push_back(Point3f(0.12, 2.0, 0.36));
  pts_3d_L.push_back(Point3f(-0.36, 2.0, 0.12));
  pts_3d_L.push_back(Point3f(-0.12, 2.0, 0.12));
  pts_3d_L.push_back(Point3f(0.12, 2.0, 0.12));
  pts_3d_L.push_back(Point3f(0.36, 2.0, 0.12));
  pts_3d_L.push_back(Point3f(-0.36, 2.0, -0.12));
  pts_3d_L.push_back(Point3f(-0.12, 2.0, -0.12));
  pts_3d_L.push_back(Point3f(0.12, 2.0, -0.12));
  pts_3d_L.push_back(Point3f(0.36, 2.0, -0.12));
  pts_3d_L.push_back(Point3f(-0.12, 2.0, -0.36));
  pts_3d_L.push_back(Point3f(0.12, 2.0, -0.36));

  pts_3d_B.push_back(Point3f(-3.0, -0.12, 0.36));
  pts_3d_B.push_back(Point3f(-3.0, 0.12, 0.36));
  pts_3d_B.push_back(Point3f(-3.0, -0.36, 0.12));
  pts_3d_B.push_back(Point3f(-3.0, -0.12, 0.12));
  pts_3d_B.push_back(Point3f(-3.0, 0.12, 0.12));
  pts_3d_B.push_back(Point3f(-3.0, 0.36, 0.12));
  pts_3d_B.push_back(Point3f(-3.0, -0.36, -0.12));
  pts_3d_B.push_back(Point3f(-3.0, -0.12, -0.12));
  pts_3d_B.push_back(Point3f(-3.0, 0.12, -0.12));
  pts_3d_B.push_back(Point3f(-3.0, 0.36, -0.12));
  pts_3d_B.push_back(Point3f(-3.0, -0.12, -0.36));
  pts_3d_B.push_back(Point3f(-3.0, 0.12, -0.36));

  pts_3d_R.push_back(Point3f(0.12, -2.0, 0.36));
  pts_3d_R.push_back(Point3f(-0.12, -2.0, 0.36));
  pts_3d_R.push_back(Point3f(0.36, -2.0, 0.12));
  pts_3d_R.push_back(Point3f(0.12, -2.0, 0.12));
  pts_3d_R.push_back(Point3f(-0.12, -2.0, 0.12));
  pts_3d_R.push_back(Point3f(-0.36, -2.0, 0.12));
  pts_3d_R.push_back(Point3f(0.36, -2.0, -0.12));
  pts_3d_R.push_back(Point3f(0.12, -2.0, -0.12));
  pts_3d_R.push_back(Point3f(-0.12, -2.0, -0.12));
  pts_3d_R.push_back(Point3f(-0.36, -2.0, -0.12));
  pts_3d_R.push_back(Point3f(0.12, -2.0, -0.36));
  pts_3d_R.push_back(Point3f(-0.12, -2.0, -0.36));

  return;
}

// 识别到的标定板的角点像素坐标
void Optimizer::Initialize2dPoints()
{
  pts_2d_F.push_back(cv::Point2f(708, 765));
  pts_2d_F.push_back(cv::Point2f(802, 765));
  pts_2d_F.push_back(cv::Point2f(615, 859));
  pts_2d_F.push_back(cv::Point2f(708, 859));
  pts_2d_F.push_back(cv::Point2f(802, 859));
  pts_2d_F.push_back(cv::Point2f(896, 859));
  pts_2d_F.push_back(cv::Point2f(615, 952));
  pts_2d_F.push_back(cv::Point2f(709, 952));
  pts_2d_F.push_back(cv::Point2f(802, 952));
  pts_2d_F.push_back(cv::Point2f(896, 952));
  pts_2d_F.push_back(cv::Point2f(708, 1046));
  pts_2d_F.push_back(cv::Point2f(802, 1046));

  pts_2d_L.push_back(cv::Point2f(703, 771));
  pts_2d_L.push_back(cv::Point2f(796, 771));
  pts_2d_L.push_back(cv::Point2f(609, 864));
  pts_2d_L.push_back(cv::Point2f(703, 864));
  pts_2d_L.push_back(cv::Point2f(796, 864));
  pts_2d_L.push_back(cv::Point2f(890, 864));
  pts_2d_L.push_back(cv::Point2f(609, 958));
  pts_2d_L.push_back(cv::Point2f(703, 958));
  pts_2d_L.push_back(cv::Point2f(796, 958));
  pts_2d_L.push_back(cv::Point2f(890, 958));
  pts_2d_L.push_back(cv::Point2f(703, 1051));
  pts_2d_L.push_back(cv::Point2f(796, 1051));

  pts_2d_B.push_back(cv::Point2f(697, 765));
  pts_2d_B.push_back(cv::Point2f(791, 765));
  pts_2d_B.push_back(cv::Point2f(603, 859));
  pts_2d_B.push_back(cv::Point2f(697, 859));
  pts_2d_B.push_back(cv::Point2f(791, 859));
  pts_2d_B.push_back(cv::Point2f(885, 859));
  pts_2d_B.push_back(cv::Point2f(603, 953));
  pts_2d_B.push_back(cv::Point2f(697, 953));
  pts_2d_B.push_back(cv::Point2f(791, 953));
  pts_2d_B.push_back(cv::Point2f(885, 953));
  pts_2d_B.push_back(cv::Point2f(697, 1047));
  pts_2d_B.push_back(cv::Point2f(791, 1047));

  pts_2d_R.push_back(cv::Point2f(702, 759));
  pts_2d_R.push_back(cv::Point2f(797, 759));
  pts_2d_R.push_back(cv::Point2f(609, 853));
  pts_2d_R.push_back(cv::Point2f(702, 853));
  pts_2d_R.push_back(cv::Point2f(797, 853));
  pts_2d_R.push_back(cv::Point2f(890, 853));
  pts_2d_R.push_back(cv::Point2f(609, 944));
  pts_2d_R.push_back(cv::Point2f(703, 944));
  pts_2d_R.push_back(cv::Point2f(796, 944));
  pts_2d_R.push_back(cv::Point2f(890, 944));
  pts_2d_R.push_back(cv::Point2f(703, 1041));
  pts_2d_R.push_back(cv::Point2f(796, 1041));

  return;
}

Optimizer::Optimizer(const Mat *imgf, const Mat *imgl, const Mat *imgb,
                     const Mat *imgr, int camera_model_index, int rows,
                     int cols, string fixed_, int flag, string data_set)
{
  // 转换成灰度图像
  imgf_rgb = *imgf;
  imgf_gray = gray_gamma(imgf_rgb);

  imgl_rgb = *imgl;
  imgl_gray = gray_gamma(imgl_rgb);

  imgb_rgb = *imgb;
  imgb_gray = gray_gamma(imgb_rgb);

  imgr_rgb = *imgr;
  imgr_gray = gray_gamma(imgr_rgb);

  brows = rows;
  bcols = cols;

  data_index = data_set; // common

  initializeK(); // 内参
  initializeD(); // 畸变
  Initialize3dPoints();
  Initialize2dPoints();
  initializePose();     // 外参
  initializeKG();       // 虚拟BEV相机内参
  initializeHeight();   // 虚拟BEV相机高度
  initializetailsize(); // 裁切尺寸

  camera_model = camera_model_index;

  bestVal_.resize(3, vector<double>(6)); // euler angles(3) and t parameters(3)

  fixed = fixed_; // front

  coarse_flag = flag;

  if (fixed == "front")
  {
    imgf_bev = project_on_ground(imgf_gray, extrinsic_front, intrinsic_front,
                                 distortion_params_front, KG, brows, bcols, hf);
    imgf_bev_rgb =
        project_on_ground(imgf_rgb, extrinsic_front, intrinsic_front,
                          distortion_params_front, KG, brows, bcols, hf);
    imgf_bev = tail(imgf_bev, "f");
    imgf_bev_rgb = tail(imgf_bev_rgb, "f");
  }
  else
  {
    imgb_bev =
        project_on_ground(imgb_gray, extrinsic_behind, intrinsic_behind,
                          distortion_params_behind, KG, brows, bcols, hb);
    imgb_bev_rgb =
        project_on_ground(imgb_rgb, extrinsic_behind, intrinsic_behind,
                          distortion_params_behind, KG, brows, bcols, hb);
    imgb_bev = tail(imgb_bev, "b");
    imgb_bev_rgb = tail(imgb_bev_rgb, "b");
  }
}

Optimizer::~Optimizer() {}

Mat Optimizer::project_on_ground(Mat img, Eigen::Matrix4d T_CG,
                                 Eigen::Matrix3d K_C, vector<double> D_C,
                                 Eigen::Matrix3d K_G, int rows, int cols,
                                 float height)
{
  Mat p_G = Mat::ones(3, rows * cols, CV_64FC1); // 定义一个BEV图像的像素齐次坐标
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      p_G.at<double>(0, cols * i + j) = j;
      p_G.at<double>(1, cols * i + j) = i;
    }
  }

  Mat P_G = Mat::ones(4, rows * cols, CV_64FC1);
  P_G(Rect(0, 0, rows * cols, 3)) = eigen2mat(K_G.inverse()) * p_G * height; // 转成世界坐标系下的齐次坐标
  // P_G(Rect(0, 0, rows * cols, 3)) = eigen2mat(K_G.inverse()) * p_G;
  if (camera_model == 0)
    P_G(Rect(0, 2, rows * cols, 1)) = cv::Mat::zeros(rows * cols, 1, CV_64FC1); // 如果是鱼眼相机z坐标设为0
  Mat P_GC = Mat::zeros(4, rows * cols, CV_64FC1);
  Mat T_CG_ =
      (Mat_<double>(4, 4) << T_CG(0, 0), T_CG(0, 1), T_CG(0, 2), T_CG(0, 3),
       T_CG(1, 0), T_CG(1, 1), T_CG(1, 2), T_CG(1, 3), T_CG(2, 0), T_CG(2, 1),
       T_CG(2, 2), T_CG(2, 3), T_CG(3, 0), T_CG(3, 1), T_CG(3, 2), T_CG(3, 3));
  P_GC = T_CG_ * P_G; // 计算SVS相机坐标系下的齐次坐标

  Mat P_GC1 = Mat::zeros(1, rows * cols, CV_64FC2);
  vector<Mat> channels(2);
  split(P_GC1, channels);
  // 归一化
  channels[0] =
      P_GC(Rect(0, 0, rows * cols, 1)) / P_GC(Rect(0, 2, rows * cols, 1));
  channels[1] =
      P_GC(Rect(0, 1, rows * cols, 1)) / P_GC(Rect(0, 2, rows * cols, 1));
  // 合并为一个两通道的向量
  merge(channels, P_GC1);

  Mat p_GC = Mat::zeros(1, rows * cols, CV_64FC2);
  Mat K_C_ = (Mat_<double>(3, 3) << K_C(0, 0), K_C(0, 1), K_C(0, 2), K_C(1, 0),
              K_C(1, 1), K_C(1, 2), K_C(2, 0), K_C(2, 1), K_C(2, 2));
  // 去畸变
  if (camera_model == 0)
  {
    fisheye::distortPoints(P_GC1, p_GC, K_C_, D_C); // fisheye
  }
  else if (camera_model == 1)
  {
    distortPointsOcam(P_GC1, p_GC, K_C, D_C); // Ocam
  }
  else
  {
    distortPoints(P_GC1, p_GC, K_C); // pinhole//转为SVS像素坐标存储在p_GC
  }

  p_GC.reshape(rows, cols);               // 这行没啥用
  Mat p_GC_table = p_GC.reshape(0, rows); // 重整为rows行cols列的矩阵，通道数不变
  Mat p_GC_table_32F;
  p_GC_table.convertTo(p_GC_table_32F, CV_32FC2); // 64位转为32位

  Mat img_GC;
  remap(img, img_GC, p_GC_table_32F, Mat(), INTER_LINEAR); // 像素映射，用线性插值，将SVS图像像素通过坐标转换映射到BEV图像中
  return img_GC;
}

Mat Optimizer::generate_surround_view(Mat img_GF, Mat img_GL, Mat img_GB,
                                      Mat img_GR)
{
  Mat dst1, dst2, dst3;
  addWeighted(img_GF, 0.5, img_GL, 0.5, 3, dst1);
  addWeighted(dst1, 1.0, img_GB, 0.5, 3, dst2);
  addWeighted(dst2, 1.0, img_GR, 0.5, 3, dst3);
  return dst3;
}

void Optimizer::Calibrate_left(int search_count, double roll_ep0,
                               double roll_ep1, double pitch_ep0,
                               double pitch_ep1, double yaw_ep0, double yaw_ep1,
                               double t0_ep0, double t0_ep1, double t1_ep0,
                               double t1_ep1, double t2_ep0, double t2_ep1)
{
  vector<double> var(6, 0);
  string varName[6] = {"roll", "pitch", "yaw", "tx", "ty", "tz"};

  max_left_loss = cur_left_loss = CostFunction(var, "left");

  random_search_params(search_count, roll_ep0, roll_ep1, pitch_ep0, pitch_ep1,
                       yaw_ep0, yaw_ep1, t0_ep0, t0_ep1, t1_ep0, t1_ep1, t2_ep0,
                       t2_ep1, "left");
}

void Optimizer::Calibrate_right(int search_count, double roll_ep0,
                                double roll_ep1, double pitch_ep0,
                                double pitch_ep1, double yaw_ep0,
                                double yaw_ep1, double t0_ep0, double t0_ep1,
                                double t1_ep0, double t1_ep1, double t2_ep0,
                                double t2_ep1)
{
  vector<double> var(6, 0);
  string varName[6] = {"roll", "pitch", "yaw", "tx", "ty", "tz"};
  max_right_loss = cur_right_loss = CostFunction(var, "right");
  random_search_params(search_count, roll_ep0, roll_ep1, pitch_ep0, pitch_ep1,
                       yaw_ep0, yaw_ep1, t0_ep0, t0_ep1, t1_ep0, t1_ep1, t2_ep0,
                       t2_ep1, "right");
}

void Optimizer::Calibrate_behind(int search_count, double roll_ep0,
                                 double roll_ep1, double pitch_ep0,
                                 double pitch_ep1, double yaw_ep0,
                                 double yaw_ep1, double t0_ep0, double t0_ep1,
                                 double t1_ep0, double t1_ep1, double t2_ep0,
                                 double t2_ep1)
{
  vector<double> var(6, 0);
  string varName[6] = {"roll", "pitch", "yaw", "tx", "ty", "tz"};
  max_behind_loss = cur_behind_loss = CostFunction(var, "behind");

  random_search_params(search_count, roll_ep0, roll_ep1, pitch_ep0, pitch_ep1,
                       yaw_ep0, yaw_ep1, t0_ep0, t0_ep1, t1_ep0, t1_ep1, t2_ep0,
                       t2_ep1, "behind");
}

void Optimizer::Calibrate_front(int search_count, double roll_ep0,
                                double roll_ep1, double pitch_ep0,
                                double pitch_ep1, double yaw_ep0,
                                double yaw_ep1, double t0_ep0, double t0_ep1,
                                double t1_ep0, double t1_ep1, double t2_ep0,
                                double t2_ep1)
{
  vector<double> var(6, 0);
  string varName[6] = {"roll", "pitch", "yaw", "tx", "ty", "tz"};
  max_front_loss = cur_front_loss = CostFunction(var, "front");
  random_search_params(search_count, roll_ep0, roll_ep1, pitch_ep0, pitch_ep1,
                       yaw_ep0, yaw_ep1, t0_ep0, t0_ep1, t1_ep0, t1_ep1, t2_ep0,
                       t2_ep1, "front");
}

void Optimizer::fine_Calibrate_left(int search_count, double roll_ep0,
                                    double roll_ep1, double pitch_ep0,
                                    double pitch_ep1, double yaw_ep0,
                                    double yaw_ep1, double t0_ep0,
                                    double t0_ep1, double t1_ep0, double t1_ep1,
                                    double t2_ep0, double t2_ep1)
{
  fine_random_search_params(search_count, roll_ep0, roll_ep1, pitch_ep0,
                            pitch_ep1, yaw_ep0, yaw_ep1, t0_ep0, t0_ep1, t1_ep0,
                            t1_ep1, t2_ep0, t2_ep1, "left");
}

void Optimizer::fine_Calibrate_right(
    int search_count, double roll_ep0, double roll_ep1, double pitch_ep0,
    double pitch_ep1, double yaw_ep0, double yaw_ep1, double t0_ep0,
    double t0_ep1, double t1_ep0, double t1_ep1, double t2_ep0, double t2_ep1)
{
  fine_random_search_params(search_count, roll_ep0, roll_ep1, pitch_ep0,
                            pitch_ep1, yaw_ep0, yaw_ep1, t0_ep0, t0_ep1, t1_ep0,
                            t1_ep1, t2_ep0, t2_ep1, "right");
}

void Optimizer::fine_Calibrate_behind(
    int search_count, double roll_ep0, double roll_ep1, double pitch_ep0,
    double pitch_ep1, double yaw_ep0, double yaw_ep1, double t0_ep0,
    double t0_ep1, double t1_ep0, double t1_ep1, double t2_ep0, double t2_ep1)
{
  fine_random_search_params(search_count, roll_ep0, roll_ep1, pitch_ep0,
                            pitch_ep1, yaw_ep0, yaw_ep1, t0_ep0, t0_ep1, t1_ep0,
                            t1_ep1, t2_ep0, t2_ep1, "behind");
}

void Optimizer::fine_Calibrate_front(
    int search_count, double roll_ep0, double roll_ep1, double pitch_ep0,
    double pitch_ep1, double yaw_ep0, double yaw_ep1, double t0_ep0,
    double t0_ep1, double t1_ep0, double t1_ep1, double t2_ep0, double t2_ep1)
{
  fine_random_search_params(search_count, roll_ep0, roll_ep1, pitch_ep0,
                            pitch_ep1, yaw_ep0, yaw_ep1, t0_ep0, t0_ep1, t1_ep0,
                            t1_ep1, t2_ep0, t2_ep1, "front");
}

double Optimizer::CostFunction(const vector<double> var, string idx)
{
  double loss;
  if (idx == "right")
  {
    Eigen::Matrix4d Tr = extrinsic_right;                   // 将未优化的外参传入
    Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var); // 转换为变换矩阵
    Tr *= deltaT;
    if (fixed == "front")
      loss = back_camera_and_compute_loss(imgf_bev, imgr_gray, Tr, "fr"); // 计算的是BEV中的f和SVS中的r的共视区域的光度误差
    else                                                                  // fixed back,so rear bev pixels back projected to right camera
      loss = back_camera_and_compute_loss(imgb_bev, imgr_gray, Tr, "br");
    return loss;
  }
  else if (idx == "left")
  {
    Eigen::Matrix4d Tl = extrinsic_left;
    Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);
    Tl *= deltaT;
    if (fixed == "front")
      loss = back_camera_and_compute_loss(imgf_bev, imgl_gray, Tl, "fl");
    else // fixed back,so rear bev pixels back projected to left camera
      loss = back_camera_and_compute_loss(imgb_bev, imgl_gray, Tl, "bl");
    return loss;
  }
  else
  { // behind(fist_order="front") or front(fist_order="behind")
    Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);
    if (fixed == "front")
    {
      Eigen::Matrix4d Tb = extrinsic_behind;
      Tb *= deltaT;
      loss = back_camera_and_compute_loss(imgl_bev, imgb_gray, Tb, "lb");
      loss += back_camera_and_compute_loss(imgr_bev, imgb_gray, Tb, "rb");
    }
    else
    { // fixed back,so left&right bev pixels back projected to front
      // camera at last
      Eigen::Matrix4d Tf = extrinsic_front;
      Tf *= deltaT;
      loss = back_camera_and_compute_loss(imgl_bev, imgf_gray, Tf, "lf");
      loss += back_camera_and_compute_loss(imgr_bev, imgf_gray, Tf, "rf");
      // cout<<loss<<endl;
    }
    return loss;
  }
}

double Optimizer::fine_CostFunction(const vector<double> var, string idx)
{
  double loss;
  if (idx == "right")
  {
    Eigen::Matrix4d Tr = extrinsic_right_opt; // 实时更新外参
    Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);
    Tr *= deltaT;
    if (fixed == "front")
      loss = back_camera_and_compute_loss(imgf_bev, imgr_gray, Tr, "fr");
    else // fixed back,so rear bev pixels back projected to right camera
      loss = back_camera_and_compute_loss(imgb_bev, imgr_gray, Tr, "br");
    return loss;
  }
  else if (idx == "left")
  {
    Eigen::Matrix4d Tl = extrinsic_left_opt;
    Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);
    Tl *= deltaT;
    if (fixed == "front")
      loss = back_camera_and_compute_loss(imgf_bev, imgl_gray, Tl, "fl");
    else // fixed back,so rear bev pixels back projected to left camera
      loss = back_camera_and_compute_loss(imgb_bev, imgl_gray, Tl, "bl");
    return loss;
  }
  else
  { // behind(fist_order="front") or front(fist_order="behind")
    Eigen::Matrix4d deltaT = TransformUtil::GetDeltaT(var);
    if (fixed == "front")
    {
      Eigen::Matrix4d Tb = extrinsic_behind_opt;
      Tb *= deltaT;
      loss = back_camera_and_compute_loss(imgl_bev, imgb_gray, Tb, "lb");
      loss += back_camera_and_compute_loss(imgr_bev, imgb_gray, Tb, "rb");
    }
    else
    { // fixed back,so left&right bev pixels back projected to front
      // camera at last
      Eigen::Matrix4d Tf = extrinsic_front_opt;
      Tf *= deltaT;
      loss = back_camera_and_compute_loss(imgl_bev, imgf_gray, Tf, "lf");
      loss += back_camera_and_compute_loss(imgr_bev, imgf_gray, Tf, "rf");
    }
    return loss;
  }
}

double Optimizer::back_camera_and_compute_loss(Mat img1, Mat img2,
                                               Eigen::Matrix4d T, string idx)
{
  vector<Point> pixels;
  Eigen::Matrix3d KC;
  vector<double> DC;
  Mat pG;
  Mat PG;
  Mat show;
  double ncoef;
  if (idx == "fl")
  { // bev_front->camera_left
    // show=imgl_rgb.clone();  //img that show back projected pixels
    DC = distortion_params_left; // distortortion params whose camera is back
                                 // projected to
    KC = intrinsic_left;         // K whose camera is back projected to
    pG = pG_fl;                  // front bev texture pixels(Mat)
    PG = PG_fl;                  // front bev pixels->3d points(Mat)
    pixels = fl_pixels_texture;  // front bev texture pixels(vector)
    ncoef = ncoef_fl;            // mean gray luminosity of front bev in commonview / mean
                                 // gray luminosity of left bev in commonview
  }
  else if (idx == "fr")
  {
    // show=imgr_rgb.clone();
    DC = distortion_params_right;
    KC = intrinsic_right;
    pG = pG_fr;
    PG = PG_fr;
    pixels = fr_pixels_texture;
    ncoef = ncoef_fr;
  }
  else if (idx == "lb")
  {
    // show=imgb_rgb.clone();
    DC = distortion_params_behind;
    KC = intrinsic_behind;
    pG = pG_bl;
    PG = PG_bl;
    pixels = bl_pixels_texture;
    ncoef = ncoef_bl;
  }
  else if (idx == "rb")
  {
    // show=imgb_rgb.clone();
    DC = distortion_params_behind;
    KC = intrinsic_behind;
    pG = pG_br;
    PG = PG_br;
    pixels = br_pixels_texture;
    ncoef = ncoef_br;
  }
  else if (idx == "lf")
  {
    // show=imgf_rgb.clone();
    DC = distortion_params_front;
    KC = intrinsic_front;
    pG = pG_fl;
    PG = PG_fl;
    pixels = fl_pixels_texture;
    ncoef = ncoef_fl;
  }
  else if (idx == "rf")
  {
    // show=imgf_rgb.clone();
    DC = distortion_params_front;
    KC = intrinsic_front;
    pG = pG_fr;
    PG = PG_fr;
    pixels = fr_pixels_texture;
    ncoef = ncoef_fr;
  }
  else if (idx == "br")
  {
    // show=imgr_rgb.clone();
    DC = distortion_params_right;
    KC = intrinsic_right;
    pG = pG_br;
    PG = PG_br;
    pixels = br_pixels_texture;
    ncoef = ncoef_br;
  }
  else if (idx == "bl")
  {
    // show=imgl_rgb.clone();
    DC = distortion_params_left;
    KC = intrinsic_left;
    pG = pG_bl;
    PG = PG_bl;
    pixels = bl_pixels_texture;
    ncoef = ncoef_bl;
  }
  // 将共视区的纹理点PG再转到相机坐标系下pG2C
  int size = pixels.size();
  if (camera_model == 0)
    PG(Rect(0, 2, size, 1)) = cv::Mat::zeros(size, 1, CV_64FC1);
  double loss = 0;
  int failcount = 0;
  Mat PG2C = Mat::zeros(4, size, CV_64FC1);
  PG2C = eigen2mat(T) * PG;
  Mat PG2C1 = Mat::zeros(1, size, CV_64FC2);
  vector<Mat> channels(2);
  split(PG2C1, channels);
  channels[0] = PG2C(Rect(0, 0, size, 1)) / PG2C(Rect(0, 2, size, 1));
  channels[1] = PG2C(Rect(0, 1, size, 1)) / PG2C(Rect(0, 2, size, 1));
  merge(channels, PG2C1);
  Mat pG2C(1, size, CV_64FC2);
  if (camera_model == 0)
    fisheye::distortPoints(PG2C1, pG2C, eigen2mat(KC), DC);
  else if (camera_model == 1)
    distortPointsOcam(PG2C1, pG2C, KC, DC);
  else
    distortPoints(PG2C1, pG2C, KC);
  for (int i = 0; i < size; i++)
  {
    double x = pG.at<double>(0, i);
    double y = pG.at<double>(1, i);
    double x1 = pG2C.at<Vec2d>(0, i)[0];
    double y1 = pG2C.at<Vec2d>(0, i)[1];
    // cout<<x1<<" "<<y1<<endl;
    if (x1 > 0 && y1 > 0 && x1 < img2.cols && y1 < img2.rows) // 累加计算BEV下纹理点和SVS下理论上是同一个点的光度误差的2范数作为loss
    {
      loss += pow(fabs(getPixelValue(&img1, x, y) / ncoef -
                       getPixelValue(&img2, x1, y1)),
                  2);
    }
    else
    {
      failcount++;
      if (failcount > 30)
        return INT_MAX;
    }
  }
  return loss;
}

void Optimizer::random_search_params(int search_count, double roll_ep0,
                                     double roll_ep1, double pitch_ep0,
                                     double pitch_ep1, double yaw_ep0,
                                     double yaw_ep1, double t0_ep0,
                                     double t0_ep1, double t1_ep0,
                                     double t1_ep1, double t2_ep0,
                                     double t2_ep1, string idx)
{
  vector<double> var(6, 0.0);
  double resolution_r = 100; // 缩放因子
  double resolution_t = 100;

  random_device generator;
  std::uniform_int_distribution<int> distribution_roll(roll_ep0 * resolution_r,
                                                       roll_ep1 * resolution_r);
  std::uniform_int_distribution<int> distribution_pitch(
      pitch_ep0 * resolution_r, pitch_ep1 * resolution_r);
  std::uniform_int_distribution<int> distribution_yaw(yaw_ep0 * resolution_r,
                                                      yaw_ep1 * resolution_r);
  std::uniform_int_distribution<int> distribution_x(t0_ep0 * resolution_t,
                                                    t0_ep1 * resolution_t);
  std::uniform_int_distribution<int> distribution_y(t1_ep0 * resolution_t,
                                                    t1_ep1 * resolution_t);
  std::uniform_int_distribution<int> distribution_z(t2_ep0 * resolution_t,
                                                    t2_ep1 * resolution_t);

  for (size_t i = 0; i < search_count; i++)
  {
    mutexval.lock();
    var[0] = double(distribution_roll(generator)) / resolution_r; // 随机生成个var(6)
    var[1] = double(distribution_pitch(generator)) / resolution_r;
    var[2] = double(distribution_yaw(generator)) / resolution_r;
    var[3] = double(distribution_x(generator)) / resolution_t;
    var[4] = double(distribution_y(generator)) / resolution_t;
    var[5] = double(distribution_z(generator)) / resolution_t;
    mutexval.unlock();

    double loss_new = CostFunction(var, idx);      // 计算损失
    if (idx == "left" && loss_new < cur_left_loss) // 迭代更新loss和优化过程中的外参
    {
      lock_guard<std::mutex> lock(mutexleft);
      cur_left_loss = loss_new;
      extrinsic_left_opt = extrinsic_left * TransformUtil::GetDeltaT(var); // 仅改变var更新优化外参
      bestVal_[0] = var;                                                   // 保存当前var(6)
    }
    if (idx == "right" && loss_new < cur_right_loss)
    {
      lock_guard<std::mutex> lock(mutexright);
      cur_right_loss = loss_new;
      extrinsic_right_opt = extrinsic_right * TransformUtil::GetDeltaT(var);
      bestVal_[1] = var;
    }
    if (idx == "behind" && loss_new < cur_behind_loss)
    {
      lock_guard<std::mutex> lock(mutexbehind);
      cur_behind_loss = loss_new;
      extrinsic_behind_opt = extrinsic_behind * TransformUtil::GetDeltaT(var);
      bestVal_[2] = var;
    }
    if (idx == "front" && loss_new < cur_front_loss)
    { // if fix back
      // camera,front camera is
      // calibrated at last
      lock_guard<std::mutex> lock(mutexfront);
      cur_front_loss = loss_new;
      extrinsic_front_opt = extrinsic_front * TransformUtil::GetDeltaT(var);
      bestVal_[2] = var;
    }
  }
  // 用优化后的外参重新投影生成BEV
  if (idx == "left")
  {
    imgl_bev = project_on_ground(imgl_gray, extrinsic_left_opt, intrinsic_left,
                                 distortion_params_left, KG, brows, bcols, hl);
    imgl_bev_rgb =
        project_on_ground(imgl_rgb, extrinsic_left_opt, intrinsic_left,
                          distortion_params_left, KG, brows, bcols, hl);
    imgl_bev = tail(imgl_bev, "l");
    imgl_bev_rgb = tail(imgl_bev_rgb, "l");
  }
  else if (idx == "right")
  {
    imgr_bev =
        project_on_ground(imgr_gray, extrinsic_right_opt, intrinsic_right,
                          distortion_params_right, KG, brows, bcols, hr);
    imgr_bev_rgb =
        project_on_ground(imgr_rgb, extrinsic_right_opt, intrinsic_right,
                          distortion_params_right, KG, brows, bcols, hr);
    imgr_bev = tail(imgr_bev, "r");
    imgr_bev_rgb = tail(imgr_bev_rgb, "r");
  }
  else if (idx == "behind")
  {
    imgb_bev =
        project_on_ground(imgb_gray, extrinsic_behind_opt, intrinsic_behind,
                          distortion_params_behind, KG, brows, bcols, hb);
    imgb_bev_rgb =
        project_on_ground(imgb_rgb, extrinsic_behind_opt, intrinsic_behind,
                          distortion_params_behind, KG, brows, bcols, hb);
    imgb_bev = tail(imgb_bev, "b");
    imgb_bev_rgb = tail(imgb_bev_rgb, "b");
  }
  else
  { // if fix back camera,front camera is calibrated at last
    imgf_bev =
        project_on_ground(imgf_gray, extrinsic_front_opt, intrinsic_front,
                          distortion_params_front, KG, brows, bcols, hf);
    imgf_bev_rgb =
        project_on_ground(imgf_rgb, extrinsic_front_opt, intrinsic_front,
                          distortion_params_front, KG, brows, bcols, hb);
    imgf_bev = tail(imgf_bev, "f");
    imgf_bev_rgb = tail(imgf_bev_rgb, "f");
  }
}

void Optimizer::fine_random_search_params(int search_count, double roll_ep0,
                                          double roll_ep1, double pitch_ep0,
                                          double pitch_ep1, double yaw_ep0,
                                          double yaw_ep1, double t0_ep0,
                                          double t0_ep1, double t1_ep0,
                                          double t1_ep1, double t2_ep0,
                                          double t2_ep1, string idx)
{
  vector<double> var(6, 0.0);

  random_device generator;
  std::uniform_real_distribution<double> distribution_roll(roll_ep0, roll_ep1);
  std::uniform_real_distribution<double> distribution_pitch(pitch_ep0,
                                                            pitch_ep1);
  std::uniform_real_distribution<double> distribution_yaw(yaw_ep0, yaw_ep1);
  std::uniform_real_distribution<double> distribution_x(t0_ep0, t0_ep1);
  std::uniform_real_distribution<double> distribution_y(t1_ep0, t1_ep1);
  std::uniform_real_distribution<double> distribution_z(t2_ep0, t2_ep1);

  if (!coarse_flag)
  {
    extrinsic_left_opt = extrinsic_left;
    extrinsic_right_opt = extrinsic_right;
    extrinsic_behind_opt = extrinsic_behind;
    extrinsic_front_opt = extrinsic_front;
  }

  for (size_t i = 0; i < search_count; i++)
  {
    mutexval.lock();
    var[0] = double(distribution_roll(generator));
    var[1] = double(distribution_pitch(generator));
    var[2] = double(distribution_yaw(generator));
    var[3] = double(distribution_x(generator));
    var[4] = double(distribution_y(generator));
    var[5] = double(distribution_z(generator));
    mutexval.unlock();
    double loss_new = fine_CostFunction(var, idx);
    if (idx == "left" && loss_new < cur_left_loss)
    {
      lock_guard<std::mutex> lock(mutexleft);
      cur_left_loss = loss_new;
      extrinsic_left_opt = extrinsic_left_opt * TransformUtil::GetDeltaT(var); // 实时更新了参与迭代的外参
      for (int i = 0; i < 6; i++)                                              // 实时更新当前外参相对于初始错误外参的变化量
      {
        bestVal_[0][i] += var[i];
      }
    }
    if (idx == "right" && loss_new < cur_right_loss)
    {
      lock_guard<std::mutex> lock(mutexright);
      cur_right_loss = loss_new;
      extrinsic_right_opt = extrinsic_right_opt * TransformUtil::GetDeltaT(var);
      for (int i = 0; i < 6; i++)
      {
        bestVal_[1][i] += var[i];
      }
    }
    if (idx == "behind" && loss_new < cur_behind_loss)
    {
      lock_guard<std::mutex> lock(mutexbehind);
      cur_behind_loss = loss_new;
      extrinsic_behind_opt =
          extrinsic_behind_opt * TransformUtil::GetDeltaT(var);
      for (int i = 0; i < 6; i++)
      {
        bestVal_[2][i] += var[i];
      }
    }
    if (idx == "front" && loss_new < cur_front_loss)
    { // if fix back
      // camera,front camera is
      // calibrated at last
      lock_guard<std::mutex> lock(mutexfront);
      cur_front_loss = loss_new;
      extrinsic_front_opt = extrinsic_front_opt * TransformUtil::GetDeltaT(var);
      for (int i = 0; i < 6; i++)
      {
        bestVal_[2][i] += var[i];
      }
    }
  }

  if (idx == "left")
  {
    imgl_bev = project_on_ground(imgl_gray, extrinsic_left_opt, intrinsic_left,
                                 distortion_params_left, KG, brows, bcols, hl);
    imgl_bev_rgb =
        project_on_ground(imgl_rgb, extrinsic_left_opt, intrinsic_left,
                          distortion_params_left, KG, brows, bcols, hl);
    imgl_bev = tail(imgl_bev, "l");
    imgl_bev_rgb = tail(imgl_bev_rgb, "l");
  }
  else if (idx == "right")
  {
    imgr_bev =
        project_on_ground(imgr_gray, extrinsic_right_opt, intrinsic_right,
                          distortion_params_right, KG, brows, bcols, hr);
    imgr_bev_rgb =
        project_on_ground(imgr_rgb, extrinsic_right_opt, intrinsic_right,
                          distortion_params_right, KG, brows, bcols, hr);
    imgr_bev = tail(imgr_bev, "r");
    imgr_bev_rgb = tail(imgr_bev_rgb, "r");
  }
  else if (idx == "behind")
  {
    imgb_bev =
        project_on_ground(imgb_gray, extrinsic_behind_opt, intrinsic_behind,
                          distortion_params_behind, KG, brows, bcols, hb);
    imgb_bev_rgb =
        project_on_ground(imgb_rgb, extrinsic_behind_opt, intrinsic_behind,
                          distortion_params_behind, KG, brows, bcols, hb);
    imgb_bev = tail(imgb_bev, "b");
    imgb_bev_rgb = tail(imgb_bev_rgb, "b");
  }
  else
  { // if fix back camera,front camera is calibrated at last
    imgf_bev =
        project_on_ground(imgf_gray, extrinsic_front_opt, intrinsic_front,
                          distortion_params_front, KG, brows, bcols, hf);
    imgf_bev_rgb =
        project_on_ground(imgf_rgb, extrinsic_front_opt, intrinsic_front,
                          distortion_params_front, KG, brows, bcols, hb);
    imgf_bev = tail(imgf_bev, "f");
    imgf_bev_rgb = tail(imgf_bev_rgb, "f");
  }
}