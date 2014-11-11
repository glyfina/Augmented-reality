#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <cstdio>
#include <GL/gl.h>
#include <GL/glut.h>
using namespace std;
using namespace cv;
#define DIFF 0.1f



#define REP(i , a) for (int i = 0 ; i < a ; i++ )

vector <Mat> rvecs, tvecs;
bool keys_compare (Point2f i,Point2f j) { return (i.y<j.y); }
void print_top_3d ( vector <Point2f> & pts , int n );
void render_bg ( const Mat& image );
GLfloat* mat_to_glf ( const Mat& m );
void drawStaticGL();
void init_ogl ();
void generateProjectionModelview ( const Mat& calibration, const Mat& rotation, const Mat& translation, Mat& projection, Mat& modelview );
int augmentation ( Mat& image , Mat & proj , Mat & mod );
void try_this ();
void filter(Mat& image);
void keyboard_func ( unsigned char key, int x, int y );
void my_gl ();
static double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<Point2f> >& imagePoints,
                                         const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                         const Mat& cameraMatrix , const Mat& distCoeffs);//,
                                         //vector<float>& perViewErrors)

void calc_cam ( FILE * in , Mat H , int w , int h );
void print_top_3d ( vector <Point2f> & pts , int n );
void read_key ( vector < pair < Point2f , Point3f > > & d2d , vector < pair < Point2f , vector <int> > > & kp   );
int match ( vector < pair < Point2f , Point3f > > & d2d , vector < pair < Point2f , vector <int> > > & kp , vector < pair < Point2f , Point3f > > & gp,vector < vector <Point2f> > & img,vector < vector <Point3f> > & obj );

const char ** gargv;

Mat image;

Mat rvec,tvec;

extern Mat thresh;

const float zNear = 0.1;			// Distance to the OpenGL near clipping plane.
const float zFar = 1000.0;

Mat det_d;
Mat det_q;
Mat image_q;
Mat image_d;
Mat dist_mat;
Mat cam_mat;
Mat det_q1;
Mat det_d1;

float u = -0.630200;
float l = 0.234000;
float z = -6.326349;


vector <Point3f> vert , nor;
vector <pair <Point3d , Point3d > > faces;

Mat rotation;						
Mat translation;				
Mat rotationMatrix;

void render_bg ( const Mat& image )
{
        // Make sure that the polygon mode is set so we draw the polygons filled
        // (save the state first so we can restore it).
        GLint polygonMode[2];
        glGetIntegerv ( GL_POLYGON_MODE, polygonMode );
        glPolygonMode ( GL_FRONT, GL_FILL );
        glPolygonMode ( GL_BACK, GL_FILL );

        // Set up the virtual camera, projecting using simple ortho so we can draw the background image.
        glMatrixMode ( GL_PROJECTION );
        glLoadIdentity();
        gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );

        glMatrixMode ( GL_MODELVIEW );
        glLoadIdentity();

        // Create a texture (on the first pass only, we will reuse it later) to hold the image we captured.
        static bool textureGenerated = false;
        static GLuint textureId;
        if ( !textureGenerated ) {
                glGenTextures ( 1, &textureId );

                glBindTexture ( GL_TEXTURE_2D, textureId );
                glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
                glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
                glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
                glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

                textureGenerated = true;
        }

        // Copy the image to the texture.
        glBindTexture ( GL_TEXTURE_2D, textureId );
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA, image.size().width, image.size().height, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, image.data );

        // Draw the image.
        glEnable ( GL_TEXTURE_2D );
        glBegin ( GL_TRIANGLES );
        glNormal3f ( 0.0, 0.0, 1.0 );

        glTexCoord2f ( 0.0, 1.0 );
        glVertex3f ( 0.0, 0.0, 0.0 );
        glTexCoord2f ( 0.0, 0.0 );
        glVertex3f ( 0.0, 1.0, 0.0 );
        glTexCoord2f ( 1.0, 1.0 );
        glVertex3f ( 1.0, 0.0, 0.0 );

        glTexCoord2f ( 1.0, 1.0 );
        glVertex3f ( 1.0, 0.0, 0.0 );
        glTexCoord2f ( 0.0, 0.0 );
        glVertex3f ( 0.0, 1.0, 0.0 );
        glTexCoord2f ( 1.0, 0.0 );
        glVertex3f ( 1.0, 1.0, 0.0 );
        glEnd();
        glDisable ( GL_TEXTURE_2D );

        // Clear the depth buffer so the texture forms the background.
        glClear ( GL_DEPTH_BUFFER_BIT );

        // Restore the polygon mode state.
        glPolygonMode ( GL_FRONT, polygonMode[0] );
        glPolygonMode ( GL_BACK, polygonMode[1] );
}

GLfloat* mat_to_glf ( const Mat& m )
{
        typedef double precision;

        Size s = m.size();
        GLfloat* mGL = new GLfloat[s.width * s.height];

        for ( int ix = 0; ix < s.width; ix++ ) {
                for ( int iy = 0; iy < s.height; iy++ ) {
                        mGL[ix * s.height + iy] = m.at<precision> ( iy, ix );
                }
        }

        return mGL;
}
void loadObj( const char *fname)
{
   FILE *fp;
   int read;
   GLfloat x, y, z;
   int a , b , c , d , e , f , g , h , i;
   char ch[3];
   fp=fopen(fname,"r");
   if (!fp)
  {
    printf("can't open file %s\n", fname);
    exit(1);
  }
   {
    while(!(feof(fp)))
    {
     read=fscanf(fp,"%s",ch);
     if(read!=0&&ch[0]=='v'&&ch[1]=='\0')
    {
     //glVertex3f(x,y,z);
     read=fscanf(fp,"%f %f %f",&x,&y,&z);
     vert.push_back (Point3f(x , y , z));
     }if(read!=0&&ch[0]=='v'&&ch[1]=='n')
    {
     //glVertex3f(x,y,z);
     read=fscanf(fp,"%f %f %f",&x,&y,&z);
     nor.push_back (Point3f(x , y , z));
     }if(read!=0&&ch[0]=='v'&&ch[1]=='t')
    {
     //glVertex3f(x,y,z);
     read=fscanf(fp,"%f %f %f",&x,&y,&z);
     //nor.push_back (Point3f(x , y , z));
     }if(read!=0&&ch[0]=='f'&&ch[1]=='\0')
    {
     //glVertex3f(x,y,z);
     read=fscanf(fp,"%d/%d/%d %d/%d/%d %d/%d/%d" , &a , &d , &g , &b , &e , &h , &c , &f , &i);
     faces.push_back (make_pair ( Point3d (a-1 , b-1 , c-1) , Point3d (g-1 , h-1 , i-1)  ));
     }
   }
   }
   LOGGER ( nor.size() );
   LOGGER ( vert.size() );
   LOGGER ( faces.size() );
   fclose(fp);
}

void drawStaticGL()
{
       // Set the colour for all new objects.
        glColor3f ( 1.0, 0.88, 0.16 );

        // The material properties for the teapot.
           
        
        
        
        float teapotAlpha = 1.0;
        float ambientTeapot[4]  = {0.3, 0.3, 1.0, teapotAlpha};
        float diffuseTeapot[4]  = {1.0, 0.0, 0.0, teapotAlpha};
        float specularTeapot[4] = {0.0, 1.0, 0.5, teapotAlpha};
        float shininessTeapot = 1;

        glMaterialfv ( GL_FRONT, GL_AMBIENT, ambientTeapot );
        glMaterialfv ( GL_FRONT, GL_DIFFUSE, diffuseTeapot );
        glMaterialfv ( GL_FRONT, GL_SPECULAR, specularTeapot );
        glMaterialf ( GL_FRONT, GL_SHININESS, shininessTeapot );

        // Draw the teapot.
        glPushMatrix();
        
        
        
        
        //glScalef ( 0.0001 , 0.0001 , 0.0001 );
       glScalef ( 0.0002 , 0.0002 , 0.0002 );
        //glTranslated (  )
        glRotatef ( 50, 0.0, 01.0, 0.0 );
        
        for(size_t i = 0; i < faces.size(); i++) {
        	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); 
		glBegin(GL_TRIANGLES); 
		//glDisable(GL_CULL_FACE);
        	//glBegin (GL_POINTS);
        	glNormal3f(nor[faces[i].second.x].x, nor[faces[i].second.x].y,nor[faces[i].second.x].z);
        	glVertex3f(vert[faces[i].first.x].x,vert[faces[i].first.x].y, vert[faces[i].first.x].z);
        	glNormal3f(nor[faces[i].second.y].x, nor[faces[i].second.y].y,nor[faces[i].second.y].z);
        	glVertex3f(vert[faces[i].first.y].x,vert[faces[i].first.y].y, vert[faces[i].first.y].z);
        	glNormal3f(nor[faces[i].second.z].x, nor[faces[i].second.z].y,nor[faces[i].second.z].z);
        	glVertex3f(vert[faces[i].first.z].x,vert[faces[i].first.z].y, vert[faces[i].first.z].z);
        	
        	glEnd();
        }
        
        glPopMatrix();
}

void init_ogl ()
{
       glClearColor ( 1.0, 1.0, 1.0, 1.0 );
	loadObj ( "sample.obj" );
        glMatrixMode ( GL_PROJECTION );
        glLoadIdentity();
        glMatrixMode ( GL_MODELVIEW );
        glLoadIdentity();

        glShadeModel ( GL_SMOOTH );
        glEnable ( GL_DEPTH_TEST );
        glEnable ( GL_LIGHTING );
        glEnable ( GL_NORMALIZE );

        glEnable ( GL_BLEND );
        glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
        // Initialise Texture States
        glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );
        glTexEnvf ( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
}

void generateProjectionModelview ( const Mat& calibration, const Mat& rotation, const Mat& translation, Mat& projection, Mat& modelview )
{
        typedef double precision;

        projection.at<precision> ( 0, 0 ) = 2 * calibration.at<precision> ( 0, 0 ) / 1632;
        projection.at<precision> ( 1, 0 ) = 0;
        projection.at<precision> ( 2, 0 ) = 0;
        projection.at<precision> ( 3, 0 ) = 0;

        projection.at<precision> ( 0, 1 ) = 0;
        projection.at<precision> ( 1, 1 ) = 2 * calibration.at<precision> ( 1, 1 ) / 1222;
        projection.at<precision> ( 2, 1 ) = 0;
        projection.at<precision> ( 3, 1 ) = 0;

        projection.at<precision> ( 0, 2 ) = 1 - 2 * calibration.at<precision> ( 0, 2 ) / 1632;
        projection.at<precision> ( 1, 2 ) = -1 + ( 2* calibration.at<precision> ( 1, 2 ) + 2 ) / 1222;
        projection.at<precision> ( 2, 2 ) = ( zNear + zFar ) / ( zNear - zFar );
        projection.at<precision> ( 3, 2 ) = -1;

        projection.at<precision> ( 0, 3 ) = 0;
        projection.at<precision> ( 1, 3 ) = 0;
        projection.at<precision> ( 2, 3 ) = 2 * zNear * zFar / ( zNear - zFar );
        projection.at<precision> ( 3, 3 ) = 0;


        modelview.at<precision> ( 0, 0 ) = rotation.at<precision> ( 0, 0 );
        modelview.at<precision> ( 1, 0 ) = rotation.at<precision> ( 1, 0 );
        modelview.at<precision> ( 2, 0 ) = rotation.at<precision> ( 2, 0 );
        modelview.at<precision> ( 3, 0 ) = 0;

        modelview.at<precision> ( 0, 1 ) = rotation.at<precision> ( 0, 1 );
        modelview.at<precision> ( 1, 1 ) = rotation.at<precision> ( 1, 1 );
        modelview.at<precision> ( 2, 1 ) = rotation.at<precision> ( 2, 1 );
        modelview.at<precision> ( 3, 1 ) = 0;

        modelview.at<precision> ( 0, 2 ) = rotation.at<precision> ( 0, 2 );
        modelview.at<precision> ( 1, 2 ) = rotation.at<precision> ( 1, 2 );
        modelview.at<precision> ( 2, 2 ) = rotation.at<precision> ( 2, 2 );
        modelview.at<precision> ( 3, 2 ) = 0;

        modelview.at<precision> ( 0, 3 ) = translation.at<precision> ( 0, 0 );
        modelview.at<precision> ( 1, 3 ) = translation.at<precision> ( 1, 0 );
        modelview.at<precision> ( 2, 3 ) = translation.at<precision> ( 2, 0 );
        modelview.at<precision> ( 3, 3 ) = 1;

        // This matrix corresponds to the change of coordinate systems.
        static double changeCoordArray[4][4] = {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0,- 1, 0}, {0, 0, 0, 1}};
        static Mat changeCoord ( 4, 4, CV_64FC1, changeCoordArray );

        modelview = changeCoord * modelview;
}

void keyboard_func ( unsigned char key, int x, int y )
{
        switch ( key ) {
        case 'q' :
        case 27 :
        	printf ( "u = %lf , l = %lf , z = %lf\n" , u , l , z);
                exit ( 0 );
                break;
        case 's' : l -= 0.1;
                printf ( "u = %lf , l = %lf , z = %lf\n" , u , l , z);
        	break;
        case 'a' : u += 0.1;
        	printf ( "u = %lf , l = %lf , z = %lf\n" , u , l , z);
        	break;
        case 'w' : l += 0.1;
        	printf ( "u = %lf , l = %lf , z = %lf\n" , u , l , z);
        	break;
        case 'd' : u -= 0.1;
        	printf ( "u = %lf , l = %lf , z = %lf\n" , u , l , z);
        	break;
        case 'z' : z += 0.1;
        	printf ( "u = %lf , l = %lf , z = %lf\n" , u , l , z);
        	break;
        case 'x' : z -= 0.1;
        	printf ( "u = %lf , l = %lf , z = %lf\n" , u , l , z);
        	break;
        }
        glutPostRedisplay();
}

int augmentation ( Mat& image , Mat & proj , Mat & mod )
{
        proj.create ( 4, 4, CV_64FC1 );
        mod.create ( 4, 4, CV_64FC1 );
        translation = tvec.clone ();
        rotation = rvec;
                
        Rodrigues ( rotation, rotationMatrix );
        
        double offsetA[3][1] = {{u}, {l}, {z}};
        Mat offset ( 3, 1, CV_64FC1, offsetA );
        translation = translation + rotationMatrix * offset;
        
        
        generateProjectionModelview ( cam_mat, rotationMatrix, translation, proj, mod );
        return 1;
}

void try_this ()
{
        
        Mat proj;
        Mat mod;
        Mat thresh;
        
        
        glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

       
       render_bg ( image_q );
       augmentation ( image_q , proj , mod );
       

        glMatrixMode ( GL_PROJECTION );
        GLfloat* projection = mat_to_glf ( proj );
        glLoadMatrixf ( projection );
        delete[] projection;

        glMatrixMode ( GL_MODELVIEW );
        GLfloat* modelview = mat_to_glf ( mod );
        glLoadMatrixf ( modelview );
        delete[] modelview;

       
        GLenum lightSource = GL_LIGHT0;
        float ambientLight[4] = {1.0, 1.0, 1.0, 1.0};
        float diffuseLight[4] = {1.0, 1.0, 1.0, 1.0};
        float specularLight[4] = {1.0, 1.0, 1.0, 1.0};
        const float lightPosition[4] = {0.0, 0.0, 1.0, 1.0};

        glEnable ( lightSource );
        glLightfv ( lightSource, GL_AMBIENT, ambientLight );
        glLightfv ( lightSource, GL_DIFFUSE, diffuseLight );
        glLightfv ( lightSource, GL_SPECULAR, specularLight );

        glPushMatrix();
        glLoadIdentity();
        glLightfv ( lightSource, GL_POSITION, lightPosition );
        glPopMatrix();

        
        
        drawStaticGL();
                
        glFlush();
        glutSwapBuffers();
        glutPostRedisplay();
       
}





void my_gl ()
{
        int a = 0;
        glutInitWindowSize ( 1632, 1222 );
        string win_name = "CVIT Heritage Project";
        glutInit ( &a, 0 );
        glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA );
        
        glutCreateWindow ( win_name.c_str() );
        
        init_ogl();
        glutDisplayFunc ( try_this );
        
        glutKeyboardFunc ( keyboard_func );
        glutIdleFunc(try_this);
	
}


static double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<Point2f> >& imagePoints,
                                         const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                         const Mat& cameraMatrix , const Mat& distCoeffs)
                                         
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    //perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); ++i )
    {
        projectPoints( Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                       distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

        int n = (int)objectPoints[i].size();
        //perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void read_key ( vector < pair < Point2f , Point3f > > & d2d , vector <KeyPoint > & kp_q , vector<KeyPoint> & kp_d , Mat& det_q, Mat& det_d  )
{
/////////////////////////////////  read from bundler files ////////////////////////////////////////////////////////





/////////////////////////////////////////////// read from key files generated from david lowe binary //////////////////////////////

        FILE*fp = fopen ( gargv[3], "r" );
        double x, y, a, b, c;
        //while ( ! feof ( fp ) )
        vector<vector<int > > det_1;
        int r , col , z;
        fscanf ( fp , "%d %d" , &r , &col );
        for ( int i = 0 ; i < r ; i++ ) {
                vector < int > dat_q;
                fscanf ( fp , "%lf %lf %lf %lf" , &y , &x , &a , &b );
                for ( int j = 0 ; j < col ; j++ ) {
                        fscanf ( fp , "%d" , &z );
                        dat_q.push_back ( z );
                }
                kp_q.push_back (  KeyPoint ( x , y, a ) );
                det_1.push_back ( dat_q );

        }
        fclose ( fp );
        //det_d.push_back(det);

        det_q = Mat ( det_1.size() , 128 , CV_32F );

        for ( size_t i = 0; i < det_1.size(); i++ ) {
                for ( size_t j = 0; j < 128; j++ ) {
                        det_q.at<float> ( i, j ) = det_1[i][j];
                }
        }


        ////////////////////////////////////////////////////////////////////////////////////////////

        fp = fopen ( gargv[4], "r" );

        
        vector<vector<int > > det_2;

        fscanf ( fp , "%d %d" , &r , &col );
        for ( int i = 0 ; i < r ; i++ ) {
                vector < int > dat_d;
                fscanf ( fp , "%lf %lf %lf %lf" , &y , &x , &a , &b );
                for ( int j = 0 ; j < col ; j++ ) {
                        fscanf ( fp , "%d" , &z );
                        dat_d.push_back ( z );
                }
                kp_d.push_back ( KeyPoint ( x , y , a ) );
                det_2.push_back ( dat_d );

        }
        fclose ( fp );
        

        det_d = Mat ( det_2.size() , 128 , CV_32F );

        for ( size_t i = 0; i < det_2.size(); i++ ) {
                for ( size_t j = 0; j < 128; j++ ) {
                        det_d.at<float> ( i, j ) = det_2[i][j];
                }
        }



        //////////////////////////////////////////////////////////////////////////////////////////////
        cout << "det_d" << det_d.rows << endl;
        cout << "det_d" << det_d.cols << endl;
        // cout<<"cols"<<kp_q.cols<<endl;
        cout << "det _ rows" << det_2.size() << endl;


}

/////////////////////////////find whether 2d points in key files match with 2d points in bundler////////////////////////////////

int match ( vector < pair < Point2f , Point3f > > & d2d , vector <  KeyPoint > & kp_q , vector < KeyPoint > & kp_d , vector < vector <Point2f> > & img, vector < vector <Point3f> > & obj ,vector < vector <Point2f> > & hom, Mat& det_q, Mat& det_d )
{


         FlannBasedMatcher matcher;
        vector<DMatch> matches;
        matcher.match ( det_q, det_d, matches );
        
        SiftDescriptorExtractor ext;
        ext.compute(image_q,kp_q,det_q1);
        ext.compute(image_d,kp_d,det_d1);
        
        
        double min_dist=100;
	double max_dist=0;
	
	for( int i = 0; i < det_q.rows; i++ )
  { double dist = matches[i].distance;
  
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );
  

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < det_q.rows; i++ )
  { if( matches[i].distance < 2*min_dist )
     { good_matches.push_back( matches[i]);
       //cout<<"id of matches in query image"<<good_matches[i].queryIdx<<endl;
       //cout<<"id of matches in train image"<<good_matches[i].trainIdx<<endl;
      }
  }
        
        
        
        
         vector<DMatch> matches2;
   vector<vector<DMatch> > m;
   
   const float minratio= 1.0f/1.5f;
   matcher.knnMatch(det_q1,det_d1,m,2);
   for(size_t i=0;i<m.size();i++){
   DMatch& bestmatch = m[i][0];
   DMatch& bettermatch=m[i][1];
   
   float distanceRatio =bestmatch.distance/bettermatch.distance;
   cout<<"dist"<<distanceRatio<<endl;
   if(distanceRatio<minratio)
   {matches2.push_back(bestmatch);}
   }
     cout<<"matches2 "<<matches2.size()<<endl;
	
        Mat img_matches;
	cv::drawMatches(image_q,kp_q,image_d,kp_d,good_matches,img_matches);
        resize(img_matches, img_matches, Size(640,480), INTER_LINEAR );
        cv::imshow("matches",img_matches);
        
        waitKey(1000);
        
       double x,y,a,b,c;
        
        vector <Point2f> qtrain;
        vector <Point2f> qquery;
        vector <Point2f> dtrain;
        vector <Point2f> dquery;
        int count=0;
        for ( size_t i = 0; i < good_matches.size(); i++ ) {
	   FILE*fp = fopen ( gargv[5], "r" );
              while ( !feof ( fp ) )
                
         {
             fscanf ( fp , "%lf %lf %lf %lf %lf" , &x , &y , &a , &b , &c );
                if (  abs(x - kp_d[good_matches[i].trainIdx ].pt.x )<0.1  &&  abs( y - kp_d[good_matches[i].trainIdx ].pt.y )<0.1  ) {

                        img[0].push_back ( Point2f ( kp_q[good_matches[i].queryIdx ].pt.x, kp_q[good_matches[i].queryIdx ].pt.y ) );
                       obj[0].push_back ( Point3f ( a, b, c ) );
                       count++;
                      
               }
               }
        }
        cout<<"count"<<count<<endl;
        cout<<"img"<<img.size()<<endl;
        cout<<"obj"<<obj.size()<<endl;
       
       
       
        fclose(fp);
return good_matches.size();
}

int main ( int argc , const char * argv [] )
{

	
    gargv = argv;
	
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	///////////////////////////////////////// calibrate ///////////////////////////////////////////////////////
	
 vector < pair < Point2f , Point3f > > d2d3 ;
        std::vector <KeyPoint > kp_q;
        std::vector <KeyPoint > kp_d;
        vector < vector <Point2f> > img ( 1 );
        vector < vector <Point3f> > obj ( 1 );
        vector < vector <Point2f> > hom ( 1 );


        image_q = imread ( gargv[1] );
        image_d = imread (gargv[2]);


        int w = 1632;
        int h = 1222;

        read_key ( d2d3 , kp_q , kp_d, det_q, det_d );
        cout << "detv" << det_q.rows << endl;
        cerr << "Match : " << match ( d2d3 , kp_q , kp_d, img, obj, hom,det_q, det_d ) << endl;
        cerr << "Size d2d3 : " << d2d3.size() << endl
             << "Size kp : " << kp_q.size() << endl;
	

	

	
 cam_mat = Mat(3 , 3 , CV_32F);
	cam_mat.at<float>(0 , 0) = w;
	cam_mat.at<float>(0 , 1) = 0;
	cam_mat.at<float>(0 , 2) = w/2;
	cam_mat.at<float>(1 , 0) = 0;
	cam_mat.at<float>(1 , 1) = w;
	cam_mat.at<float>(1 , 2) = h/2;
	cam_mat.at<float>(2 , 0) = 0;
	cam_mat.at<float>(2 , 1) = 0;
	cam_mat.at<float>(2 , 2) = 1;
	
	double rms = calibrateCamera ( obj, img, Size (w , h), cam_mat, dist_mat, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS );
	cout<<"cam_mat"<<cam_mat<<endl;
        cout<<"rms without : "<<rms<<endl;	
	
	Mat r , t;
	solvePnP ( obj[0] , img[0] , cam_mat , dist_mat , r , t );
	//cout << "R: " << r << endl
	//	<< "T: " << t << endl;
	rvec = r;
	tvec = t;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////		
	
	
	
	my_gl ();
	
        glutMainLoop ();
	return 0;
}
