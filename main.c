#include <stdlib.h>
#include <stdio.h>
#define OPENCLWRAPPER_IMPLEMENTATION
#include "OpenCLWrapper.h"
#define PID 0
#define DID 0
#include "common.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <float.h>

//TODO: Normalize units
//TODO: Convergence condition of dt
//TODO: Share buffers between OpenCL and OpenGL to reduce overhead by read/write

typedef struct {
	float x;
	float y;
	float z;
	float u;
	float v;
} vertex;

typedef struct {
	double max_psi2;
	double integral;
	double max_pot;
	double min_pot;
} return_data;


typedef struct {
	float x;
	float y;
	float z;
	float w;
} vec4f;

int width = 800;
int height = 800;
double mousex = 0;
double mousey = 0;

//https://github.com/tsoding/ded/blob/master/src/main.c
void MessageCallback(GLenum source,
                     GLenum type,
                     GLuint id,
                     GLenum severity,
                     GLsizei length,
                     const GLchar* message,
                     const void* userParam) {
    (void) source;
    (void) id;
    (void) length;
    (void) userParam;
    fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
            (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
            type, severity, message);
}

void resize_callback(GLFWwindow* window, int width_, int height_) {
	(void)window;
	width = width_;
	height = height_;
    glViewport(0, 0, width_, height_);
}

void initGL(GLFWwindow **window) {
    if (!glfwInit()) {
        fprintf(stderr, "Couldn't init GLFW\n");
        exit(1);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    *window = glfwCreateWindow(width, height, "Scho", NULL, NULL);

    if (!*window) {
        fprintf(stderr, "Couldn't create window\n");
        glfwTerminate();
        exit(1);
    }
    glfwMakeContextCurrent(*window);

    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Couldn't init GLEW\n");
        glfwTerminate();
        exit(1);
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);    

    glfwGetCursorPos(*window, &mousex, &mousey);
    glfwSetFramebufferSizeCallback(*window, resize_callback);
    if (GLEW_ARB_debug_output) {
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(MessageCallback, 0);
    } else {
        fprintf(stderr, "WARNING! GLEW_ARB_debug_output is not available");
    }
}

unsigned int initShader(const char* file_path, unsigned int type)
{
    unsigned int id = glCreateShader(type);
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Could not open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    int file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* data = (char*)malloc(file_size + 1);
    fread(data, file_size, 1, file);
    data[file_size] = '\0';

    glShaderSource(id, 1, (const char* const*)&data, &file_size);
    glCompileShader(id);

    int success;

    glGetShaderiv(id, GL_COMPILE_STATUS, &success);

    if (success)
        printf("Shader %u compiled\n", id);
    else
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* info = (char*)malloc(length);
        glGetShaderInfoLog(id, length, &length, info);
        printf("Build log: \n%s\n", info);
        free(info);
    }

    free(data);
    fclose(file);
	return id;
}

return_data max_and_integrate(const double *norm2, const double* pot, int total, double dx, double dy) {
	return_data ret = {0};
	ret.max_psi2 = norm2[0];
	ret.max_pot = pot[0];
	ret.min_pot = pot[0];
	for (int i = 0; i < total; ++i) {
		ret.max_psi2 = norm2[i] > ret.max_psi2? norm2[i]: ret.max_psi2;
		ret.integral += norm2[i] * dx * dy;
		ret.max_pot = pot[i] > ret.max_pot? pot[i]: ret.max_pot;
		ret.min_pot = pot[i] < ret.min_pot? pot[i]: ret.min_pot;
	}
	ret.integral = sqrt(ret.integral);
	return ret;
}

int main() {
	///////////////////////////////////////////////////////////////////////////
	double mass		= 1.0 * ME;
	double x0		= -1.0 * a0;
	double x1		= 1.0 * a0;
	double y0		= -1.0 * a0;
	double y1		= 1.0 * a0;
	double lx		= x1 - x0;
	double ly		= y1 - y0;
	int ncols		= 272;
	int nrows		= 272;
	double dx		= lx / ncols;
	double dy		= ly / nrows;
	complex *psi0	= (complex*)calloc(nrows * ncols, sizeof(complex));
	complex *psi	= (complex*)calloc(nrows * ncols, sizeof(complex));
	double *norm2	= (double*)calloc(nrows * ncols, sizeof(double));
	double *pot		= (double*)calloc(nrows * ncols, sizeof(double));
	vec4f *tex		= (vec4f*)calloc(nrows * ncols, sizeof(vec4f));
	vec4f *pot_tex	= (vec4f*)calloc(nrows * ncols, sizeof(vec4f));

	double dt = 0.01 * HBAR / E0;

	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			double x = (x0 + j * dx) / a0;
			double y = (y0 + i * dy) / a0;
			psi0[i * ncols + j] = cmul((complex){exp((-x * x - y * y) / 0.01), 0.0}, cexp_((x + y) / 0.01));
			psi[i * ncols + j] = psi0[i * ncols + j];
			norm2[i * ncols + j] = cmul(psi0[i * ncols + j], ccon(psi0[i * ncols + j])).r;
		}
	}
	return_data integral = max_and_integrate(norm2, pot, nrows * ncols, dx, dy);
	for (int i = 0; i < nrows * ncols; ++i) {
		psi0[i] = cmul((complex){1.0 / integral.integral, 0.0}, psi0[i]);
		psi[i] = psi0[i];
	}

	///////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////////////////////////////////////////
	size_t n_plats;
	cl_platform_id *p_ids = InitPlatforms(&n_plats);

	size_t n_devs;
	cl_device_id *d_ids = InitDevices(p_ids[PID], &n_devs);

	for (size_t i = 0; i < n_plats; ++i) PlatformInfo(stdout, p_ids[i], i);
	for (size_t i = 0; i < n_devs; ++i) DeviceInfo(stdout, d_ids[i], i);

	cl_context ctx = InitContext(d_ids, n_devs);
	cl_command_queue queue = InitQueue(ctx, d_ids[DID]);
	char *kernel;
	ReadFile("./kernel.c", &kernel);
	cl_program program = InitProgramSource(ctx, (const char*)kernel);
	cl_int build_error = BuildProgram(program, n_devs, d_ids, "-DOPENCL_COMP");
	BuildProgramInfo(stdout, program, d_ids[DID], build_error);

	const char* funcs[] = {"Step", "Normalize", "psi2", "to_rgba", "Att"};
	Kernel *kernels = InitKernels(program, funcs, 5);

	cl_mem dpsi0	= CreateBuffer(sizeof(complex) * nrows * ncols, ctx, CL_MEM_READ_WRITE);
	cl_mem dpsi		= CreateBuffer(sizeof(complex) * nrows * ncols, ctx, CL_MEM_READ_WRITE);
	cl_mem dnorm2	= CreateBuffer(sizeof(double) * nrows * ncols, ctx, CL_MEM_READ_WRITE);
	cl_mem dtex		= CreateBuffer(sizeof(vec4f) * nrows * ncols, ctx, CL_MEM_READ_WRITE);
	cl_mem dpot		= CreateBuffer(sizeof(double) * nrows * ncols, ctx, CL_MEM_READ_WRITE);
	cl_mem dpot_tex = CreateBuffer(sizeof(vec4f) * nrows * ncols, ctx, CL_MEM_READ_WRITE);

	WriteBuffer(dpsi0, psi0, sizeof(complex) * nrows * ncols, 0, queue);
	WriteBuffer(dpsi, psi, sizeof(complex) * nrows * ncols, 0, queue);

	size_t global_work	= nrows * ncols;
	size_t local_work	= 32;



	SetKernelArg(kernels[0], 0, sizeof(int), &nrows);
	SetKernelArg(kernels[0], 1, sizeof(int), &ncols);
	SetKernelArg(kernels[0], 2, sizeof(double), &mass);
	SetKernelArg(kernels[0], 3, sizeof(double), &dx);
	SetKernelArg(kernels[0], 4, sizeof(double), &dy);
	//ARG 6 -> time
	SetKernelArg(kernels[0], 7, sizeof(cl_mem), &dpsi);
	SetKernelArg(kernels[0], 8, sizeof(cl_mem), &dpsi0);
	SetKernelArg(kernels[0], 9, sizeof(double), &x0);
	SetKernelArg(kernels[0], 10, sizeof(double), &y0);
	SetKernelArg(kernels[0], 11, sizeof(cl_mem), &dpot);


	SetKernelArg(kernels[1], 0, sizeof(cl_mem), &dpsi);
	//ARG 1 -> norm


	SetKernelArg(kernels[2], 0, sizeof(cl_mem), &dpsi);
	SetKernelArg(kernels[2], 1, sizeof(cl_mem), &dnorm2);


	SetKernelArg(kernels[3], 0, sizeof(cl_mem), &dpsi0);
	SetKernelArg(kernels[3], 1, sizeof(cl_mem), &dtex);
	//ARG 2 -> max_norm2
	SetKernelArg(kernels[3], 3, sizeof(cl_mem), &dpot);
	// ARG 4, 5 MAX MIN POT
	SetKernelArg(kernels[3], 6, sizeof(cl_mem), &dpot_tex);

	SetKernelArg(kernels[4], 0, sizeof(cl_mem), &dpsi);
	SetKernelArg(kernels[4], 1, sizeof(cl_mem), &dpsi0);
	
	 
	
	///////////////////////////////////////////////////////////////////////////

	
	///////////////////////////////////////////////////////////////////////////
	//GL
	GLFWwindow *window = NULL;
	initGL(&window);
	vertex vert_psi[] = {{-1.0f, -1.0f, 0.0f, 0.0f, 0.0f},
						 { 1.0f, -1.0f, 0.0f, 1.0f, 0.0f},
						 { 1.0f,  1.0f, 0.0f, 1.0f, 1.0f},
						 {-1.0f,  1.0f, 0.0f, 0.0f, 1.0f}};
	unsigned int ind_psi[] = {0, 1, 2, 0, 2, 3};

	unsigned int VAO_psi, VBO_psi, EBO_psi;

    glGenVertexArrays(1, &VAO_psi);
    glGenBuffers(1, &VBO_psi);
    glGenBuffers(1, &EBO_psi);
    glBindVertexArray(VAO_psi);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_psi);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vert_psi), vert_psi, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_psi);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ind_psi), ind_psi, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, u));
    glEnableVertexAttribArray(1);	

    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0);




	vertex vert_pot[] = {{-1.0f, -1.0f, 0.5f, 0.0f, 0.0f},
						 { 1.0f, -1.0f, 0.5f, 1.0f, 0.0f},
						 { 1.0f,  1.0f, 0.5f, 1.0f, 1.0f},
						 {-1.0f,  1.0f, 0.5f, 0.0f, 1.0f}};
	unsigned int ind_pot[] = {0, 1, 2, 0, 2, 3};

	unsigned int VAO_pot, VBO_pot, EBO_pot;

    glGenVertexArrays(1, &VAO_pot);
    glGenBuffers(1, &VBO_pot);
    glGenBuffers(1, &EBO_pot);
    glBindVertexArray(VAO_pot);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_pot);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vert_pot), vert_pot, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_pot);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ind_pot), ind_pot, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, u));
    glEnableVertexAttribArray(1);	

    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0);




    unsigned int tex_psi;
    glGenTextures(1, &tex_psi);
    glBindTexture(GL_TEXTURE_2D, tex_psi); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ncols, nrows, 0, GL_RGBA, GL_FLOAT, tex);
	glBindTexture(GL_TEXTURE_2D, 0);

    unsigned int tex_potential;
    glGenTextures(1, &tex_potential);
    glBindTexture(GL_TEXTURE_2D, tex_potential); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ncols, nrows, 0, GL_RGBA, GL_FLOAT, pot_tex);
	glBindTexture(GL_TEXTURE_2D, 0);


	unsigned int frag_id = initShader("./frag.glsl", GL_FRAGMENT_SHADER);
	unsigned int vert_id = initShader("./vertex.glsl", GL_VERTEX_SHADER);
	unsigned int prog_id = glCreateProgram();
	glAttachShader(prog_id, vert_id);
	glAttachShader(prog_id, frag_id);
	glLinkProgram(prog_id);
	glDeleteShader(vert_id);
	glDeleteShader(frag_id);
	///////////////////////////////////////////////////////////////////////////

	double last_time = glfwGetTime();
	size_t count;
	
	glUseProgram(prog_id);
    glUniform1i(glGetUniformLocation(prog_id, "tex"), 0);
	
	while (!glfwWindowShouldClose(window)) {
		double current_time = glfwGetTime();
		dt = current_time - last_time;
		last_time = current_time;
		if (count % 100 == 0) printf("ms: %e\n", dt / 1.0e-3);
		
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
        // Step -> psi2 -> normalize -> to_rgba -> att
		// 0 -> 2 -> 1 -> 3 -> 4
		
		/* dt *= maxd * maxd / (2.0 * HBAR / (2.0 * mass)) * HBAR / E0; */
		dt *= 0.0001 * HBAR / E0;
		double t = count * dt;
		count++;
		t = current_time;
		
		SetKernelArg(kernels[0], 6, sizeof(double), &t);
    	SetKernelArg(kernels[0], 5, sizeof(double), &dt);
		
		EnqueueND(queue, kernels[0], 1, NULL, &global_work, &local_work);
		EnqueueND(queue, kernels[2], 1, NULL, &global_work, &local_work);
		
		ReadBuffer(dnorm2, norm2, sizeof(double) * nrows * ncols, 0, queue);
		ReadBuffer(dpot, pot, sizeof(double) * nrows * ncols, 0, queue);

		return_data integral = max_and_integrate(norm2, pot, nrows * ncols, dx, dy);
		
		SetKernelArg(kernels[1], 1, sizeof(double), &integral.integral);
		SetKernelArg(kernels[3], 2, sizeof(double), &integral.max_psi2);
		SetKernelArg(kernels[3], 4, sizeof(double), &integral.max_pot);
		SetKernelArg(kernels[3], 5, sizeof(double), &integral.min_pot);

		
		EnqueueND(queue, kernels[1], 1, NULL, &global_work, &local_work);
		EnqueueND(queue, kernels[3], 1, NULL, &global_work, &local_work);
		ReadBuffer(dtex, tex, sizeof(vec4f) * nrows * ncols, 0, queue);
		ReadBuffer(dpot_tex, pot_tex, sizeof(vec4f) * nrows * ncols, 0, queue);
		EnqueueND(queue, kernels[4], 1, NULL, &global_work, &local_work);
		
	
        glBindTexture(GL_TEXTURE_2D, tex_psi);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ncols, nrows, 0, GL_RGBA, GL_FLOAT, tex);
		
        glBindTexture(GL_TEXTURE_2D, tex_potential);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ncols, nrows, 0, GL_RGBA, GL_FLOAT, pot_tex);
		
		glBindTexture(GL_TEXTURE_2D, 0);

        glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex_potential);
		
		glBindVertexArray(VAO_pot);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);


		
        glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex_psi);

		glBindVertexArray(VAO_psi);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

			

		glfwSwapInterval(0);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

	
	
	free(p_ids);
	free(d_ids);
	
	free(psi0);
	free(psi);
	free(norm2);
	free(pot);
	free(tex);
	free(pot_tex);
	PrintCLError(stderr, clReleaseMemObject(dpsi0), "Error releasing dpsi0");
	PrintCLError(stderr, clReleaseMemObject(dpsi), "Error releasing dpsi");
	PrintCLError(stderr, clReleaseMemObject(dnorm2), "Error releasing dnorm2");
	PrintCLError(stderr, clReleaseMemObject(dtex), "Error releasing dtex");
	PrintCLError(stderr, clReleaseMemObject(dpot), "Error releasing dpot");
	PrintCLError(stderr, clReleaseMemObject(dpot_tex), "Error releasing dpot_tex");
	
	free(kernel);
	free(kernels);
    glDeleteVertexArrays(1, &VAO_psi);
    glDeleteBuffers(1, &VBO_psi);
    glDeleteBuffers(1, &EBO_psi);

    glDeleteVertexArrays(1, &VAO_pot);
    glDeleteBuffers(1, &VBO_pot);
    glDeleteBuffers(1, &EBO_pot);
	
	glfwTerminate();
	return 0;
}
