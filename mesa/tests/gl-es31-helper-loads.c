/* SPDX-License-Identifier: MIT
 *
 * Fragment helper-invocation memory loads: dFdx/dFdy of a value loaded from an
 * SSBO on a diagonal triangle edge (partially covered quads). Writes the 64x64
 * RGBA8 result to $OUT; compare against stock NVIDIA / llvmpipe byte for byte.
 * On nvrm without NVK's SET_PRIV_REG sm_disp_ctrl bit-3 clear, 32 edge pixels
 * differed (helper lanes loaded 0); with it the output equals stock 615.
 */
/* helper-invocation SSBO loads: dFdx/dFdy of an SSBO value on a diagonal triangle edge */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdio.h>
#include <stdlib.h>
static GLuint sh(GLenum t, const char *s){GLuint x=glCreateShader(t);glShaderSource(x,1,&s,0);glCompileShader(x);GLint ok;glGetShaderiv(x,GL_COMPILE_STATUS,&ok);if(!ok){char l[2048];glGetShaderInfoLog(x,2048,0,l);printf("%s\n",l);exit(1);}return x;}
int main(void){
 PFNEGLGETPLATFORMDISPLAYEXTPROC gd=(void*)eglGetProcAddress("eglGetPlatformDisplayEXT");
 EGLDisplay d=gd(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,0); eglInitialize(d,0,0); eglBindAPI(EGL_OPENGL_ES_API);
 static const EGLint a[]={EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
 EGLContext c=eglCreateContext(d,EGL_NO_CONFIG_KHR,EGL_NO_CONTEXT,a); eglMakeCurrent(d,0,0,c);
 const char*vs="#version 310 es\nvoid main(){vec2 p[3]=vec2[3](vec2(-1,-1),vec2(1,-1),vec2(-1,1));gl_Position=vec4(p[gl_VertexID],0,1);}\n";
 const char*fs="#version 310 es\nprecision highp float;precision highp int;layout(std430,binding=0) readonly buffer B{float v[];};layout(location=0) out vec4 o;\n"
  "void main(){uint i=uint(gl_FragCoord.y)*64u+uint(gl_FragCoord.x);float a=v[i];o=vec4(dFdx(a)*0.25+0.5,dFdy(a)*0.25+0.5,a*0.01,1);}\n";
 GLuint p=glCreateProgram();glAttachShader(p,sh(GL_VERTEX_SHADER,vs));glAttachShader(p,sh(GL_FRAGMENT_SHADER,fs));glLinkProgram(p);glUseProgram(p);
 static float v[64*64]; for(int y=0;y<64;y++)for(int x=0;x<64;x++)v[y*64+x]=(float)((x*x)%7+y%5)*0.5f;
 GLuint b;glGenBuffers(1,&b);glBindBuffer(GL_SHADER_STORAGE_BUFFER,b);glBufferData(GL_SHADER_STORAGE_BUFFER,sizeof v,v,GL_STATIC_DRAW);glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,b);
 GLuint t,f;glGenTextures(1,&t);glBindTexture(GL_TEXTURE_2D,t);glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,64,64);glGenFramebuffers(1,&f);glBindFramebuffer(GL_FRAMEBUFFER,f);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t,0);
 glViewport(0,0,64,64);glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);glDrawArrays(GL_TRIANGLES,0,3);
 static unsigned char px[64*64*4];glReadPixels(0,0,64,64,GL_RGBA,GL_UNSIGNED_BYTE,px);
 FILE*o=fopen(getenv("OUT"),"wb");fwrite(px,1,sizeof px,o);fclose(o);printf("%s done\n",glGetString(GL_RENDERER));return 0;}
