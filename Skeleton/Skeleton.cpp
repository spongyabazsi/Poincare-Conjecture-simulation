//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Paréj Balázs
// Neptun : E1512Q
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
using namespace std;

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers
	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel
	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";


// 2D camera
class Camera2D {
public:
	vec2 wCenter; // center in world coordinates
	vec2 wSize;   // width and height in world coordinates

	Camera2D() : wCenter(0, 0), wSize(20, 20) { }

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;
bool animate = false;
float tCurrent = 0;
GPUProgram gpuProgram;
const int nTesselatedVertices = 100;
float originalarea = 0;
vec4 originalcentro(0, 0, 0, 0);
class Szakasz {
public:
	vec4 start;
	vec4 end;
	Szakasz(vec4 s, vec4 e) {
		start = s;
		end = e;
	}
	void setSzakasz(vec4 st, vec4 e) {
		start = st; end = e;
	}

	float getLength() {
		float x = end.x - start.x;
		float y = end.y - start.y;
		float l = sqrtf(x*x + y * y);
		return l;
	}
	bool DoesItCross(Szakasz sz) {
		Szakasz inf(start, end);
		float y11 = sz.start.y;
		float y12 = sz.end.y;
		float x11 = sz.start.x;
		float x12 = sz.end.x;
		float y21 = inf.start.y;
		float y22 = inf.end.y;
		float x21 = inf.start.x;
		float x22 = inf.end.x;
		float t2 = ((y12 - y22) - ((x12 - x22) * (y21 - y22)
			/ (x21 - x22)))
			/
			(((y21 - y22) * (x11 - x12)
				/ (x21 - x22)) - (y11 - y12));
		float t1 = t2 * ((x11 - x12) + (x12 - x22)) / (x21 - x22);
		if (t1 > 0 && t1 < 1 && t2 > 0 && t2 < 1)
			return true;
		else
			return false;
	}
	bool IsItInside(Szakasz sz) {
		vec4 midpoint;
		midpoint.x = (end.x + start.x) / 2;
		midpoint.y = (end.y + start.y) / 2;
		Szakasz inf(midpoint, vec4(100, 100, 0, 0));
		float y11 = sz.start.y;
		float y12 = sz.end.y;
		float x11 = sz.start.x;
		float x12 = sz.end.x;
		float y21 = inf.start.y;
		float y22 = inf.end.y;
		float x21 = inf.start.x;
		float x22 = inf.end.y;
		float t2 = ((y12 - y22) - ((x12 - x22) * (y21 - y22)
			/ (x21 - x22)))
			/
			(((y21 - y22) * (x11 - x12)
				/ (x21 - x22)) - (y11 - y12));
		float t1 = t2 * ((x11 - x12) + (x12 - x22)) / (x21 - x22);
		if (t1 > 0 && t1 < 1 && t2 > 0 && t2 < 1)
			return true;
		else
			return false;
	}
};

unsigned int vaoCurve, vboCurve, vaocps, vbocps;
unsigned int vaoConvexHull, vboConvexHull;
unsigned int vaoCent, vboCent;
class CatmullRom {
	vector<vec4> cps;
	vector<float> knots;
	std::vector<vec4> vertexDatacpy;
	std::vector<vec4> trianglescpy;
	vector<vec4> centroids;
	float area;
	vec4 Hermite(vec4 p0, vec4 v0, float t0,
		vec4 p1, vec4 v1, float t1,
		float t) {
		vec4 a3 = ((p0 - p1) * 2.0f) / ((powf(t1 - t0, 3.0f))) + (v1 + v0) / (powf(t1 - t0, 2.0f));
		vec4 a2 = ((p1 - p0) * 3.0f) / ((powf(t1 - t0, 2.0f))) - (v1 + (v0 * 2.0f)) / ((t1 - t0));
		vec4 a1 = v0;
		vec4 a0 = p0;
		return a3 * powf(t - t0, 3.0f) + a2 * powf(t - t0, 2.0f) + a1 * (t - t0) + a0;
	}

public:
	virtual void AddControlPoint(vec2 cPoint) {
		vec4 wVertex = vec4(cPoint.x, cPoint.y, 0, 1) * camera.Pinv() * camera.Vinv();
		cps.push_back(wVertex);
		knots.push_back((float)cps.size());
	}
	vec4 r(float t) {
		vec4 res;

		for (int i = 0; i < cps.size() - 1; i++) {
			vec4 v0, v1;
			if (knots[i] <= t && t <= knots[i + 1]) {
				if (cps.size() == 2) return res = Hermite(cps[0], vec4(0, 0, 0, 0), knots[0], cps[1], vec4(0, 0, 0, 0), knots[1], t);
				else
					if (i == 0) {
						v0 = ((cps[i + 1] - cps[i]) * (1 / (knots[i + 1] - knots[i])) +
							(cps[i] - cps[cps.size() - 1]) * (1 / (knots[i] - (knots[cps.size() - 1]))))* 0.5;
						v1 = ((cps[i + 2] - cps[i + 1]) * (1 / (knots[i + 2] - knots[i + 1])) +
							(cps[i + 1] - cps[i]) * (1 / (knots[i + 1] - knots[i])))* 0.5;
						return res = Hermite(cps[i], v0, knots[i], cps[i + 1], v1, knots[i + 1], t);
					}
					else if (i == cps.size() - 2) {
						v0 = ((cps[i + 1] - cps[i]) * (1 / (knots[i + 1] - knots[i])) +
							(cps[i] - cps[i - 1]) * (1 / (knots[i] - knots[i - 1])))* 0.5;
						v1 = ((cps[0] - cps[i + 1]) * (1 / (knots[0] - knots[i + 1])) +
							(cps[i + 1] - cps[i]) * (1 / (knots[i + 1] - knots[i])))* 0.5;
						return res = Hermite(cps[i], v0, knots[i], cps[i + 1], v1, knots[i + 1], t);
					}
					else {
						v0 = ((cps[i + 1] - cps[i]) * (1 / (knots[i + 1] - knots[i])) + (cps[i] - cps[i - 1]) * (1 / (knots[i] - knots[i - 1])))* 0.5;
						i++;
						v1 = ((cps[i + 1] - cps[i]) * (1 / (knots[i + 1] - knots[i])) + (cps[i] - cps[i - 1]) * (1 / (knots[i] - knots[i - 1])))*0.5;
						i--;
						return res = Hermite(cps[i], v0, knots[i], cps[i + 1], v1, knots[i + 1], t);
					}
			}
		}
		if (t > knots[cps.size() - 1] && cps.size() > 2) {
			vec4 v0, v1;
			v0 = ((cps[0] - cps[cps.size() - 1]) * (1 / (knots[0] - knots[cps.size() - 1])) +
				(cps[cps.size() - 1] - cps[cps.size() - 2]) * (1 / (knots[cps.size() - 1] - knots[cps.size() - 2])))* 0.5;
			v1 = ((cps[1] - cps[0]) * (1 / (knots[1] - knots[0])) +
				(cps[0] - cps[cps.size() - 1]) * (1 / (knots[0] - knots[cps.size() - 1])))* 0.5;
			return res = Hermite(cps[cps.size() - 1], v0, knots[cps.size() - 1], cps[0], v1, knots[cps.size() - 1] + 1, t);
		}
	}


	float getArea() {
		centroids.clear();
		float area = 0;
		for (int i = 0; i < trianglescpy.size(); i = i + 3) {
			vec3 a(trianglescpy[i].x - trianglescpy[i + 1].x, trianglescpy[i].y - trianglescpy[i + 1].y, 0);
			vec3 b(trianglescpy[i].x - trianglescpy[i + 2].x, trianglescpy[i].y - trianglescpy[i + 2].y, 0);
			vec3 crossp = cross(a, b);
			vec4 center((trianglescpy[i].x + trianglescpy[i + 1].x + trianglescpy[i + 2].x) / 3,
				(trianglescpy[i].y + trianglescpy[i + 1].y + trianglescpy[i + 2].y) / 3,
				fabs(crossp.z*0.5), 0.0f);
			centroids.push_back(center);
			area = area + fabs(crossp.z*0.5);
		}
		this->area = area;
		return area;

	}

	vec4 Centroid() {
		float cx = 0, cy = 0;
		for (int i = 0; i < centroids.size(); i++) {
			cx = (centroids[i].x * centroids[i].z) + cx;
			cy = (centroids[i].y * centroids[i].z) + cy;
		}
		return vec4(cx / area, cy / area, 0, 0);
	}


	float tGet(int i) {
		return knots[i];
	}
	float tStart() {
		return knots[0];
	}
	float tEnd() { return knots[knots.size() - 1] + 1; }


	void create() {
		glGenVertexArrays(1, &vaoConvexHull);
		glBindVertexArray(vaoConvexHull);

		glGenBuffers(1, &vboConvexHull);
		glBindBuffer(GL_ARRAY_BUFFER, vboConvexHull);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);

		glGenVertexArrays(1, &vaoCurve);
		glBindVertexArray(vaoCurve);

		glGenBuffers(1, &vboCurve);
		glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);

		glGenVertexArrays(1, &vaoCent);
		glBindVertexArray(vaoCent);

		glGenBuffers(1, &vboCent);
		glBindBuffer(GL_ARRAY_BUFFER, vboCent);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
		glGenVertexArrays(1, &vaocps);
		glBindVertexArray(vaocps);

		glGenBuffers(1, &vbocps);
		glBindBuffer(GL_ARRAY_BUFFER, vbocps);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
	}

	vector<vec4> Triangulation(vector<vec4> vert) {
		vector<vec4> triangles;
		bool cross = false;
		int insidecounter = 0;
		for (int i = 0; i < vert.size();)
		{
			int left = 0;
			int right = 0;
			cross = false;
			if (i == 0) {
				left = vert.size() - 1; right = 1;
			}
			else if (i == vert.size() - 1) {
				left = vert.size() - 2; right = 0;
			}
			else {
				left = i - 1; right = i + 1;
			}
			Szakasz sz(vert[left], vert[right]);
			for (int j = 0; j < vert.size(); j++) {
				int current = 0;
				int next = 0;
				if (!((j == i) || (j == left) || (j == right))) {
					if (j == vert.size() - 1) {
						current = j; next = 0;
					}
					else {
						current = j; next = j + 1;
					}
					Szakasz collision(vert[current], vert[next]);
					if (sz.DoesItCross(collision))
					{
						cross = true;
					}
					if (sz.IsItInside(collision)) {
						insidecounter++;
					}
				}
			}

			if ((cross == false) && (insidecounter % 2 == 1))
			{
				triangles.push_back(vert[left]);
				triangles.push_back(vert[i]);
				triangles.push_back(vert[right]);
				vert.erase(vert.begin() + i);

				i = 0;
			}
			else { i++; }
			insidecounter = 0;
		}
		trianglescpy = triangles;
		return triangles;
	}
	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();
		gpuProgram.setUniform(VPTransform, "MVP");

		if (cps.size() > 0) {
			glBindVertexArray(vaocps);
			glBindBuffer(GL_ARRAY_BUFFER, vbocps);
			glBufferData(GL_ARRAY_BUFFER, cps.size() * 4 * sizeof(float), &cps[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(1, 0, 0), "color");
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, cps.size());
		}

		if (cps.size() >= 2) {	// draw curve
			std::vector<vec4> vertexData;
			for (int i = 0; i < nTesselatedVertices; i++) {
				float tNormalized = (float)i / (nTesselatedVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec4 wVertex = r(t);
				wVertex.z = t;
				vertexData.push_back(wVertex);
			}


			if (animate == false) {

				vector<vec4> triangles;
				triangles.clear();
				triangles = Triangulation(vertexData);
				vertexDatacpy = vertexData;
				trianglescpy = triangles;

				getArea();
				glBindVertexArray(vaoConvexHull);
				glBindBuffer(GL_ARRAY_BUFFER, vboConvexHull);
				glBufferData(GL_ARRAY_BUFFER, triangles.size() * 4 * sizeof(float), &triangles[0], GL_STATIC_DRAW);
				gpuProgram.setUniform(vec3(0, 1, 1), "color");
				glDrawArrays(GL_TRIANGLES, 0, triangles.size());


				glBindVertexArray(vaoCurve);
				glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
				glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(vec4), &vertexData[0], GL_DYNAMIC_DRAW);
				gpuProgram.setUniform(vec3(1, 1, 1), "color");
				glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);


				vec4 centro = Centroid();
				glBindVertexArray(vaoCent);
				glBindBuffer(GL_ARRAY_BUFFER, vboCent);
				glBufferData(GL_ARRAY_BUFFER, sizeof(vec4), &centro, GL_DYNAMIC_DRAW);
				gpuProgram.setUniform(vec3(0, 1, 0), "color");
				glPointSize(10.0f);
				glDrawArrays(GL_POINTS, 0, 1);
			}
			else
			{

				getArea();
				glBindVertexArray(vaoConvexHull);
				glBindBuffer(GL_ARRAY_BUFFER, vboConvexHull);
				glBufferData(GL_ARRAY_BUFFER, vertexDatacpy.size() * 4 * sizeof(float), &vertexDatacpy[0], GL_STATIC_DRAW);
				gpuProgram.setUniform(vec3(0, 0.5f, 1), "color");
				glDrawArrays(GL_POINTS, 0, vertexDatacpy.size());

				glBindVertexArray(vaoConvexHull);
				glBindBuffer(GL_ARRAY_BUFFER, vboConvexHull);
				glBufferData(GL_ARRAY_BUFFER, trianglescpy.size() * 4 * sizeof(float), &trianglescpy[0], GL_STATIC_DRAW);
				gpuProgram.setUniform(vec3(0, 1, 1), "color");
				glDrawArrays(GL_TRIANGLES, 0, trianglescpy.size());

				vec4 centro = Centroid();
				glBindVertexArray(vaoCent);
				glBindBuffer(GL_ARRAY_BUFFER, vboCent);
				glBufferData(GL_ARRAY_BUFFER, sizeof(vec4), &centro, GL_DYNAMIC_DRAW);
				gpuProgram.setUniform(vec3(0, 1, 0), "color");
				glPointSize(10.0f);
				glDrawArrays(GL_POINTS, 0, 1);

				glBindVertexArray(vaoCent);
				glBindBuffer(GL_ARRAY_BUFFER, vboCent);
				glBufferData(GL_ARRAY_BUFFER, sizeof(vec4), &originalcentro, GL_DYNAMIC_DRAW);
				gpuProgram.setUniform(vec3(0.6f, 0.0f, 0.5f), "color");
				glPointSize(10.0f);
				glDrawArrays(GL_POINTS, 0, 1);
			}
		}
	}
	void Animate(float dt) {
		float konstans = 0.01f / dt; //ez a hazibeado miatt ennyi, visual studioban 0.01/dt-vel mukodik jol
		int r1 = 0, r2 = 0, r3 = 0;
		float L1 = 0, L2 = 0, L3 = 0;
		float tau1 = 0, tau2 = 0, tau3 = 0;
		vec4 K(0, 0, 0, 0);
		for (int i = 0; i < vertexDatacpy.size(); i = i + 1) {
			if (i == 0) {
				tau2 = 0;
				r1 = vertexDatacpy.size() - 1;
				r2 = 0;
				r3 = i + 1;
				Szakasz t1(vertexDatacpy[r1], vertexDatacpy[r2]);
				Szakasz t2(vertexDatacpy[r3], vertexDatacpy[r2]);
				tau1 = -1 * t1.getLength();
				tau3 = t2.getLength();
			}
			else if (i == vertexDatacpy.size() - 1) {
				tau2 = 0;
				r1 = vertexDatacpy.size() - 2;
				r2 = i;
				r3 = 0;
				Szakasz t1(vertexDatacpy[vertexDatacpy.size() - 2], vertexDatacpy[i]);
				Szakasz t2(vertexDatacpy[0], vertexDatacpy[i]);
				tau1 = -1 * t1.getLength();
				tau3 = t2.getLength();


			}
			else {

				r1 = i - 1;
				r2 = i;
				r3 = i + 1;
				Szakasz t1(vertexDatacpy[i - 1], vertexDatacpy[i]);
				Szakasz t2(vertexDatacpy[i + 1], vertexDatacpy[i]);
				tau1 = -1 * t1.getLength();
				tau2 = 0;
				tau3 = t2.getLength();
			}


			L1 = 2 / ((tau1 - tau2)*(tau1 - tau3));
			L2 = 2 / ((tau2 - tau1)*(tau2 - tau3));
			L3 = 2 / ((tau3 - tau2)*(tau3 - tau1));

			K = vertexDatacpy[r2] * L2 + vertexDatacpy[r3] * L3 + vertexDatacpy[r1] * L1;


			if (fabs(K.x) > 24 || fabs(K.y) > 24) {
				K = vec4(0, 0, 0, 0);
			}

			vertexDatacpy[i] = vertexDatacpy[i] + K * dt * konstans;
		}
		float newarea = getArea();
		float lambda = sqrtf(originalarea / newarea);
		mat4 scale = { lambda,0.0f,0.0f,0.0f,
					  0.0f,lambda,0.0f,0.0f,
					  0.0f, 0.0f, lambda, 0.0f,
						0.0f,0.0f,0.0f, 1.0f };
		vec4 newcentro = Centroid();
		vec4 stride = -1 * (newcentro - originalcentro);
		mat4 tolo = { 1.0f,0.0f,0.0f,0.0f,
					  0.0f,1.0f,0.0f,0.0f,
					  0.0f, 0.0f, 1.0f, 0.0f,
						stride.x,stride.y,stride.z, 1.0f };
		for (int i = 0; i < vertexDatacpy.size(); i++)
		{
			vertexDatacpy[i] = vertexDatacpy[i] + stride;
			vertexDatacpy[i] = vertexDatacpy[i] * scale;
		}
		trianglescpy = Triangulation(vertexDatacpy);
	}
};

CatmullRom cr;
unsigned int vao;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	cr.create();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}


void onDisplay() {
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	cr.Draw();

	glutSwapBuffers();
}


void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case 'd': camera.Pan(vec2(-1, 0)); break;
	case 'a': animate = true; originalarea = cr.getArea(); originalcentro = cr.Centroid(); break;
	case 'z': camera.Zoom(0.9f); break;
	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}


void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		cr.AddControlPoint(vec2(cX, cY));
	}
}

void onIdle() {
	static float tend = 0;

	const float dt = 0.1;

	float tstart = tend;

	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	if (animate) for (float t = tstart; t < tend; t += dt) {

		float Dt = fmin(dt, tend - t);
		cr.Animate(Dt);

	}

	glutPostRedisplay();
}