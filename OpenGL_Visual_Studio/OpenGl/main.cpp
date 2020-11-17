#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include <map>
#include <conio.h>
#include <unordered_set>
#include <queue>

#include <windows.h>
#include <glut.h>

using namespace std;

#define pi (2*acos(0.0))
#define eps 0.000001

#define N 200
#define M 150
#define cpx 4
#define numobs 40
#define crange 40
#define osa 10
#define osd 5
#define lmax 500.0
#define inputfilename "voronoi_6.txt"


int G[N][M];
bool B[N][M];
int T;
int numvcells;
vector< vector<int> > AL;
vector<int> scl;

FILE* finv;


double max2(double a, double b)
{
    return (a > b) ? a : b;
}

double min2(double a, double b)
{
    return (a < b) ? a : b;
}


class point
{
public:
    double x, y;
    point(double xx = 0.0, double yy = 0.0)
    {
        x = xx;
        y = yy;
    }
};

class scanline
{
public:
    int rownum;
    int scol;
    int ecol;
    int cnt;
    int ub;
    int lb;
    scanline()
    {
        rownum = scol = ecol = cnt = ub = lb = -1;
    }
};


class cell
{
public:
    int x, y;
    point c;
    cell(int xx = 0, int yy = 0)
    {
        x = xx;
        y = yy;
        c.x = (x + 0.5) * cpx;
        c.y = (y + 0.5) * cpx;
    }
};

class obs
{
public:
    double xl, xh, yl, yh;
    obs(double xxl = 0.0, double xxh = 0.0, double yyl = 0.0, double yyh = 0.0)
    {
        xl = xxl;
        xh = xxh;
        yl = yyl;
        yh = yyh;
    }
};


vector<obs> obsdata;
vector<point> vors;
vector< vector<point> > allvors;
vector<point> lchain;
vector<point> rchain;
vector<scanline> slines;
vector<cell> traj;
vector< vector<cell> > alltraj;



void po(obs o)
{
    cout << o.xl << " " << o.xh << " " << o.yl << " " << o.yh << endl;
}


void ppoint(point p)
{
    cout << p.x << " " << p.y << endl;
}

bool doesint1(obs a, obs b)
{
    if (((b.xl <= a.xl && a.xl <= b.xh) || (b.xl <= a.xh && a.xh <= b.xh) || (a.xl <= b.xl && b.xl <= a.xh) || (a.xl <= b.xh && b.xh <= a.xh)) && ((b.yl <= a.yl && a.yl <= b.yh) || (b.yl <= a.yh && a.yh <= b.yh) || (a.yl <= b.yl && b.yl <= a.yh) || (a.yl <= b.yh && b.yh <= a.yh)))return true;

    return false;
}

bool doesintn(vector<obs> ol, obs o)
{
    int i;
    for (i = 0;i < ol.size();i++)if (doesint1(ol[i], o) == true)return true;

    return false;
}

bool isinside1(obs o, point p)
{
    if (o.xl < p.x + eps && p.x < o.xh + eps && o.yl < p.y + eps && p.y < o.yh + eps)return true;

    return false;
}

bool isinsiden(vector<obs> ol, point p)
{
    int i;
    for (i = 0;i < ol.size();i++)if (isinside1(ol[i], p) == true)return true;

    return false;
}


//generate obstacles uniformly over the environment at random
void genobs()
{
    obsdata.clear();

    obs o;
    int i, ri, rii;
    double rd;
    srand(time(NULL));

    int cnt = numobs;
    while (cnt != 0)
    {
        o.xl = 1 + rand() % (N - osa - osd);
        ri = rand() % (2 * osd) + (osa - osd);
        o.xh = o.xl + ri;

        o.yl = 1 + rand() % (M - osa - osd);
        ri = rand() % (2 * osd) + (osa - osd);
        o.yh = o.yl + ri;

        o.xl *= cpx;
        o.xh *= cpx;
        o.yl *= cpx;
        o.yh *= cpx;


        if (doesintn(obsdata, o) == false)
        {
            obsdata.push_back(o);
            cnt--;
        }
    }
}


//determine free (true) and occupied (false) grid cells
void genblockedcells()
{
    int i, j, k;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            point tp;
            tp.x = i * cpx + (cpx / 2.0);
            tp.y = j * cpx + (cpx / 2.0);

            bool flag = true;
            for (k = 0; k < numobs; k++)
            {
                if (isinside1(obsdata[k], tp) == true)
                {
                    flag = false;
                    break;
                }
            }
            B[i][j] = flag;
        }
    }
}


double area3p(point a, point b, point c)
{
    return a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
}

bool doesint(point a, point b, point c, point d)
{
    if (area3p(a, b, c) * area3p(a, b, d) <= 0 && area3p(c, d, a) * area3p(c, d, b) <= 0)
        return true;
    return false;
}

//determine if point p falls within any obstacle
vector<obs> rangequery(point p)
{
    int j;
    vector<obs> ret;
    obs to;
    to.xl = p.x - crange;
    to.xh = p.x + crange;
    to.yl = p.y - crange;
    to.yh = p.y + crange;
    for (j = 0;j < obsdata.size();j++)
    {
        if (doesint1(obsdata[j], to) == true)ret.push_back(obsdata[j]);
    }
    return ret;
}


void drawSquare(double x, double y, double s)
{
    glBegin(GL_QUADS); {
        glVertex3f(x + s, y + s, 0);
        glVertex3f(x + s, y - s, 0);
        glVertex3f(x - s, y - s, 0);
        glVertex3f(x - s, y + s, 0);
    }glEnd();
}


void drawCircle(double x, double y, double radius, double h)
{
    int i;
    int segments = 16;
    struct point points[17];
    //generate points
    for (i = 0;i <= segments;i++)
    {
        points[i].x = x + radius * cos(((double)i / (double)segments) * 2 * pi);
        points[i].y = y + radius * sin(((double)i / (double)segments) * 2 * pi);
    }
    //draw segments using generated points
    for (i = 0;i < segments;i++)
    {
        glBegin(GL_TRIANGLES);
        {
            glVertex3f(x, y, h);
            glVertex3f(points[i].x, points[i].y, h);
            glVertex3f(points[i + 1].x, points[i + 1].y, h);
        }
        glEnd();
    }
}


void keyboardListener(unsigned char key, int x, int y) {
    switch (key) {

    case '1':
        //drawgrid = 1 - drawgrid;
        break;

    default:
        break;
    }
}


void specialKeyListener(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_DOWN:		//down arrow key
        break;
    case GLUT_KEY_UP:		// up arrow key
        break;

    case GLUT_KEY_RIGHT:
        break;
    case GLUT_KEY_LEFT:
        break;

    case GLUT_KEY_PAGE_UP:
        break;
    case GLUT_KEY_PAGE_DOWN:
        break;

    case GLUT_KEY_INSERT:
        break;

    case GLUT_KEY_HOME:
        break;
    case GLUT_KEY_END:
        break;

    default:
        break;
    }
}


void mouseListener(int button, int state, int x, int y) {	//x, y is the x-y of the screen (2D)
    switch (button) {
    case GLUT_LEFT_BUTTON:
        if (state == GLUT_DOWN) {		// 2 times?? in ONE click? -- solution is checking DOWN or UP

        }
        break;

    case GLUT_RIGHT_BUTTON:
        //........
        break;

    case GLUT_MIDDLE_BUTTON:
        //........
        break;

    default:
        break;
    }
}





void display() {

    //clear the display
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0, 0, 0, 0);	//color black
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /********************
    / set-up camera here
    ********************/
    //load the correct matrix -- MODEL-VIEW matrix
    glMatrixMode(GL_MODELVIEW);

    //initialize the matrix
    glLoadIdentity();

    //now give three info
    //1. where is the camera (viewer)?
    //2. where is the camera looking?
    //3. Which direction is the camera's UP direction?

    //gluLookAt(100,100,100,	0,0,0,	0,0,1);

//	if (top == 0) gluLookAt(240 + 275 * cos(cameraangle), 240 + 275 * sin(cameraangle), cameraheight,		240,240,0,		0,0,1);
//	else if (top == 1) gluLookAt(0,0,0,	0,0,-1,	0,1,0);

    gluLookAt(0, 0, 0, 0, 0, -1, 0, 1, 0);


    //again select MODEL-VIEW
    glMatrixMode(GL_MODELVIEW);


    /****************************
    / Add your objects from here
    ****************************/
    //add objects


    int i, j;

    //dras obstacles in red
    glColor3f(1, 0, 0);
    for (int i = 0; i < obsdata.size(); i++)
    {
        glBegin(GL_QUADS);
        {
            glVertex3f(obsdata[i].xl, obsdata[i].yl, 0);
            glVertex3f(obsdata[i].xh, obsdata[i].yl, 0);
            glVertex3f(obsdata[i].xh, obsdata[i].yh, 0);
            glVertex3f(obsdata[i].xl, obsdata[i].yh, 0);
        }
        glEnd();
    }

    //draw ugvs as circles in blue
    glColor3f(0, 0, 1);
    for (int i = 0; i < numvcells; i++)
    {
        int sq = T % (int)alltraj[i].size();
        int x = cpx * alltraj[i][sq].x + cpx / 2.0;
        int y = cpx * alltraj[i][sq].y + cpx / 2.0;

        drawCircle(x, y, 3, 0.1);
    }



    //draw free grid cells according to latency value
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            glColor3f(0, 1 - (double)(G[i][j] / lmax), 0);
            glBegin(GL_QUADS);
            {
                glVertex3f(i * cpx, j * cpx, 0);
                glVertex3f((i + 1) * cpx, j * cpx, 0);
                glVertex3f((i + 1) * cpx, (j + 1) * cpx, 0);
                glVertex3f(i * cpx, (j + 1) * cpx, 0);
            }
            glEnd();
        }
    }

    //draw boundary of voronoi regions
    glLineWidth(2.0);
    glColor3f(1, 0, 1);
    for (int i = 0; i < (int)allvors.size(); i++)
    {
        for (int j = 0; j < (int)allvors[i].size() - 1; j++)
        {
            glBegin(GL_LINES);
            {
                glVertex3f(allvors[i][j].x, allvors[i][j].y, 0.1);
                glVertex3f(allvors[i][j + 1].x, allvors[i][j + 1].y, 0.1);
            }
            glEnd();
        }
    }

    //ADD this line in the end --- if you use double buffer (i.e. GL_DOUBLE)
    glutSwapBuffers();
}


//update position of UGVs and latency values of free grid cells at each time step
void animate(int val) {
    //codes for any changes in Models, Camera

    T++;

    unordered_set<int> st;

    for (int i = 0; i < numvcells; i++)
    {
        int sq = T % (int)alltraj[i].size();
        int x = alltraj[i][sq].x;
        int y = alltraj[i][sq].y;

        int cn = y * N + x;

        for (int j = 0; j < AL[cn].size(); j++) st.insert(AL[cn][j]);
    }

    for (int i = 0; i < M * N; i++)
    {
        auto it = st.find(i);
        if (it != st.end()) G[i % N][i / N] = 0;
        else if (G[i % N][i / N] < lmax) G[i % N][i / N]++;
    }

    glutPostRedisplay();
    glutTimerFunc(35, animate, 0);
}




//find the cells visible from a given cell
vector<int> processcell(cell c)
{
    point p = c.c;
    vector<int> ret;
    ret.clear();

    //determine bounding square around cells visible from a UGV
    vector<obs> vo = rangequery(p);
    int lgn = max(0, c.x - (int)(crange / cpx));
    int rgn = min(N - 1, c.x + (int)(crange / cpx));
    int bgn = max(0, c.y - (int)(crange / cpx));
    int tgn = min(M - 1, c.y + (int)(crange / cpx));


    //check each cell within the bounding square to see if its visible from the UGV
    int i, j, k;
    for (i = lgn; i <= rgn; i++)
    {
        for (j = bgn; j <= tgn; j++)
        {
            point tp = point((i + 0.5) * cpx, (j + 0.5) * cpx);
            if (B[i][j] == false) continue;
            if ((p.x - tp.x) * (p.x - tp.x) + (p.y - tp.y) * (p.y - tp.y) > crange * crange) continue;
            bool flag = true;
            for (k = 0; k < vo.size(); k++)
            {
                if (doesint(p, tp, point(vo[k].xl, vo[k].yl), point(vo[k].xl, vo[k].yh)))
                {
                    flag = false;
                    break;
                }
                if (doesint(p, tp, point(vo[k].xh, vo[k].yl), point(vo[k].xh, vo[k].yh)))
                {
                    flag = false;
                    break;
                }
                if (doesint(p, tp, point(vo[k].xl, vo[k].yl), point(vo[k].xh, vo[k].yl)))
                {
                    flag = false;
                    break;
                }
                if (doesint(p, tp, point(vo[k].xl, vo[k].yh), point(vo[k].xh, vo[k].yh)))
                {
                    flag = false;
                    break;
                }
            }
            if (flag == true)
            {
                ret.push_back(j * N + i);
            }
        }
    }
    return ret;
}


//take as input the boundary of voronoi region from file
void inputvoronoi()
{
    int psize;
    fscanf(finv, " %d", &psize);
    vors.clear();
    for (int i = 0; i < psize; i++)
    {
        point tp;
        fscanf(finv, " %lf %lf", &tp.x, &tp.y);
        vors.push_back(tp);
    }
    vector<point> tmpv = vors;
    tmpv.push_back(vors[0]);
    allvors.push_back(tmpv);
}


//get starting column of the scanline at height h within the current voronoi region
int getscol(double h)
{
    for (int i = 0; i < (int)lchain.size() - 1; i++)
    {
        if (lchain[i].y >= h && h >= lchain[i + 1].y)
        {
            double c = (lchain[i + 1].x - lchain[i].x) * (h - lchain[i].y);
            c /= (lchain[i + 1].y - lchain[i].y);
            c += lchain[i].x;
            int tmp = floor(c / (cpx / 2.0));

            if (tmp % 2 == 0) return tmp / 2;
            else return (tmp / 2) + 1;
        }
    }
}


//get ending column of the scanline at height h within the current voronoi region
int getecol(double h)
{
    for (int i = 0; i < (int)rchain.size() - 1; i++)
    {
        if (rchain[i].y >= h && h >= rchain[i + 1].y)
        {
            double c = (rchain[i + 1].x - rchain[i].x) * (h - rchain[i].y);
            c /= (rchain[i + 1].y - rchain[i].y);
            c += rchain[i].x;
            int tmp = floor(c / (cpx / 2.0));

            if (tmp % 2 == 0) return (tmp / 2) - 1;
            else return (tmp / 2);
        }
    }
}


//process one voronoi region by generating the scanline hights in the vector scl
void processvoronoicell()
{
    double maxy = -1.0;
    int maxyidx = -1;


    //determine point in voronoi region boundary with maximum y value/height
    for (int i = 0; i < vors.size(); i++)
    {
        if (vors[i].y >= maxy)
        {
            maxy = vors[i].y;
            maxyidx = i;
        }
    }
    rotate(vors.begin(), vors.begin() + maxyidx, vors.end());

    //determine left chain and right chain
    int lp = (int)vors.size() - 1;
    for (int i = 0; i < (int)vors.size() - 1; i++)
    {
        if (vors[i].y < vors[i + 1].y)
        {
            lp = i;
            break;
        }
    }

    lchain.clear();
    for (int i = 0; i <= lp; i++) lchain.push_back(vors[i]);
    if (lchain[(int)lchain.size() - 1].y == lchain[(int)lchain.size() - 2].y) lchain.pop_back();


    rchain.clear();
    rchain.push_back(vors[0]);

    for (int i = (int)vors.size() - 1; i >= lp; i--) rchain.push_back(vors[i]);
    if (rchain[0].y == rchain[1].y) rchain.erase(rchain.begin());




    //determine height/y value of topmost and bottom most scanline that intersects the voronoi region
    int topr, botr;
    double topy = vors[0].y;
    double boty = vors[lp].y;
    int tmp;

    tmp = floor(topy / (cpx / 2.0));
    if (tmp % 2 == 0) topr = (tmp / 2) - 1;
    else topr = tmp / 2;

    tmp = floor(boty / (cpx / 2.0));
    if (tmp % 2 == 0) botr = tmp / 2;
    else botr = (tmp / 2) + 1;


    slines.clear();


    //determine the starting column and ending column of all scanlines
    for (int i = botr; i <= topr; i++)
    {
        double ht = i * cpx + (cpx / 2.0);

        scanline tmp;
        int scol, ecol;

        scol = getscol(ht);
        while (B[scol][i] == false) scol++;
        tmp.scol = scol;

        ecol = getecol(ht);
        while (B[ecol][i] == false) ecol--;
        tmp.ecol = ecol;

        tmp.rownum = i;
        slines.push_back(tmp);
    }

    vector<scanline> tmps;
    tmps.clear();
    for (int i = 0; i < M; i++)
    {
        scanline slt;
        slt.cnt = 0;
        tmps.push_back(slt);
    }

    for (int i = 0; i < slines.size(); i++)
    {
        if (slines[i].scol <= slines[i].ecol)
        {
            int cnt = 0;
            for (int j = slines[i].scol; j <= slines[i].ecol; j++) if (B[j][slines[i].rownum] == true) cnt++;
            slines[i].cnt = cnt;
            tmps[slines[i].rownum] = slines[i];
        }
    }
    slines = tmps;

    while (slines[botr].cnt == 0) botr++;
    while (slines[topr].cnt == 0) topr--;


    //determine upper bound (ub) of each scanline which represents the farthest scanline above
    //such that there is no non-visible cells within current and ub scanline

    for (int y = botr; y <= topr; y++)
    {

        int ub = y + floor((2 * crange) / cpx);
        ub = min(ub, topr);


        for (int ul = ub; ul >= y; ul--)
        {
            unordered_set<int> vc;
            vc.clear();


            for (int i = slines[y].scol; i <= slines[y].ecol; i++)
            {
                if (B[i][y] == true)
                {
                    int cn = y * N + i;
                    for (int j = 0; j < AL[cn].size(); j++)
                    {
                        int xx = AL[cn][j] % N;
                        int yy = AL[cn][j] / N;
                        if (yy >= y && yy <= ul && xx >= slines[yy].scol && xx <= slines[yy].ecol) vc.insert(AL[cn][j]);
                    }
                }
            }


            for (int i = slines[ul].scol; i <= slines[ul].ecol; i++)
            {
                if (B[i][ul] == true)
                {
                    int cn = ul * N + i;
                    for (int j = 0; j < AL[cn].size(); j++)
                    {
                        int xx = AL[cn][j] % N;
                        int yy = AL[cn][j] / N;
                        if (yy <= ul && yy >= y && xx >= slines[yy].scol && xx <= slines[yy].ecol) vc.insert(AL[cn][j]);
                    }
                }
            }

            int sum = 0;
            for (int i = y; i <= ul; i++) sum += slines[i].cnt;


            if (sum == (int)vc.size())
            {
                slines[y].ub = ul;
                break;
            }
        }
    }

    //generate scl, the list of scanlines that visits all cells within the voronoi region
    scl.clear();
    scl.push_back(botr);
    while (scl.back() < topr) scl.push_back(slines[scl.back()].ub);

}


//fing unobstructed path from grid cell (xp, yp) to grid cell (xn, yn) using bfs
vector<cell> findpath(int xp, int yp, int xn, int yn)
{
    bool vis[N][M];
    for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) vis[i][j] = false;
    pair<int, int> par[N][M];
    for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) par[i][j] = pair<int, int>(-1, -1);

    queue<cell> Q;
    cell src(xn, yn);
    vis[xn][yn] = true;
    Q.push(src);

    vector< pair<int, int> > dirs = { pair<int, int>(0, 1), pair<int, int>(0, -1), pair<int, int>(1, 0), pair<int, int>(-1, 0) };
    //vector< pair<int, int> > dirs = {pair<int, int>(0, 1), pair<int, int>(0, -1), pair<int, int>(1, 0), pair<int, int>(-1, 0), pair<int, int>(1, 1), pair<int, int>(1, -1), pair<int, int>(-1, 1), pair<int, int>(-1, -1)};


    bool flag = false;

    while (true)
    {
        cell s = Q.front();
        Q.pop();
        int x = s.x;
        int y = s.y;

        random_shuffle(dirs.begin(), dirs.end());


        for (int i = 0; i < (int)dirs.size(); i++)
        {
            if (x + dirs[i].first >= N || x + dirs[i].first < 0 || y + dirs[i].second >= M || y + dirs[i].second < 0) continue;
            if (vis[x + dirs[i].first][y + dirs[i].second] == false && B[x + dirs[i].first][y + dirs[i].second] == true)
            {
                vis[x + dirs[i].first][y + dirs[i].second] = true;
                par[x + dirs[i].first][y + dirs[i].second] = pair<int, int>(x, y);
                Q.push(cell(x + dirs[i].first, y + dirs[i].second));
                if (x + dirs[i].first == xp && y + dirs[i].second == yp)
                {
                    flag = true;
                    break;
                }
            }
        }

        if (flag == true) break;

    }

    vector<cell> ret;
    ret.clear();
    int curx = xp;
    int cury = yp;

    while (!(curx == xn && cury == yn))
    {
        int tx = par[curx][cury].first;
        int ty = par[curx][cury].second;

        curx = tx;
        cury = ty;

        ret.push_back(cell(curx, cury));
    }


    ret.pop_back();
    return ret;
}


//generate the trajectory, i.e., list of adjacent grid cells that visits all scanlines in scl in a zig zag motion
void gentraj()
{
    traj.clear();


    for (int i = 0; i < scl.size(); i++)
    {
        vector<cell> tc;
        tc.clear();
        for (int j = slines[scl[i]].scol; j <= slines[scl[i]].ecol; j++)
        {
            if (B[j][scl[i]] == true) tc.push_back(cell(j, scl[i]));
            else
            {
                int x = j;
                int y = scl[i];
                while (B[x][y] == false)
                {
                    tc.push_back(cell(x - 1, y + 1));
                    y++;
                }
                while (B[x][y - 1] == false)
                {
                    tc.push_back(cell(x, y));
                    x++;
                }
                y--;
                while (y > scl[i])
                {
                    tc.push_back(cell(x, y));
                    y--;
                }
                j = x - 1;

            }
        }

        if (i % 2 == 0) traj.insert(traj.end(), tc.begin(), tc.end());
        else
        {
            reverse(tc.begin(), tc.end());
            traj.insert(traj.end(), tc.begin(), tc.end());
        }
        if (i < (int)scl.size() - 1)
        {
            vector<cell> path;
            path.clear();

            if (i % 2 == 0)
            {
                path = findpath(slines[scl[i]].ecol, scl[i], slines[scl[i + 1]].ecol, scl[i + 1]);
            }
            else
            {
                path = findpath(slines[scl[i]].scol, scl[i], slines[scl[i + 1]].scol, scl[i + 1]);
            }
            traj.insert(traj.end(), path.begin(), path.end());

        }
    }

    vector<cell> traj2 = traj;
    reverse(traj2.begin(), traj2.end());
    traj.insert(traj.begin(), traj2.begin(), traj2.end());

}


//generate the trajectories for all voronoi regions
void init() {
    //codes for initialization


    int i, j;
    T = 0;

    genobs();
    genblockedcells();


    cout << "Processing grid cells..." << endl;
    AL.clear();
    for (i = 0; i < M; i++) for (j = 0; j < N; j++) AL.push_back(processcell(cell(j, i)));
    for (i = 0; i < N; i++) for (j = 0; j < M; j++) G[i][j] = lmax;
    cout << "Grid cells processed." << endl;


    finv = fopen(inputfilename, "r");
    fscanf(finv, " %d", &numvcells);


    alltraj.clear();

    for (i = 0; i < numvcells; i++)
    {
        inputvoronoi();
        processvoronoicell();
        gentraj();
        alltraj.push_back(traj);
        cout << "Voronoi region " << i << " processed." << endl;
    }

    fclose(finv);




    //clear the screen
    glClearColor(0, 0, 0, 0);

    /************************
    / set-up projection here
    ************************/
    //load the PROJECTION matrix
    glMatrixMode(GL_PROJECTION);

    //initialize the matrix
    glLoadIdentity();

    //give PERSPECTIVE parameters
    gluOrtho2D(-10, N * cpx + 10, -10, M * cpx + 10);


    //field of view in the Y (vertically)
    //aspect ratio that determines the field of view in the X direction (horizontally)
    //near distance
    //far distance
}


int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitWindowSize(N * cpx + 20, M * cpx + 20);
    glutInitWindowPosition(0, 0);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);	//Depth, Double buffer, RGB color

    glutCreateWindow("Baseline Lawnmower");

    init();

    glEnable(GL_DEPTH_TEST);	//enable Depth Testing

    glutDisplayFunc(display);	//display callback function
    glutTimerFunc(50, animate, 0);		//what you want to do in the idle time (when no drawing is occuring)

    //glutIdleFunc(animate);

    glutKeyboardFunc(keyboardListener);
    glutSpecialFunc(specialKeyListener);
    glutMouseFunc(mouseListener);

    glutMainLoop();		//The main loop of OpenGL

    return 0;
}


