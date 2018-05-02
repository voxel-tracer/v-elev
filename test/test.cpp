// test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "pdf.h"


int main()
{
	float3 dirs[6];
	dirs[0] = make_float3(1, 0, 0);
	dirs[1] = make_float3(-1, 0, 0);
	dirs[2] = make_float3(0,1, 0);
	dirs[3] = make_float3(0,-1, 0);
	dirs[4] = make_float3(0, 0, 1);
	dirs[5] = make_float3(0, 0, -1);

	for (uint i = 0; i < 6; i++)
	{
		float3 dir = dirs[i];
		printf("(%.0f, %.0f, %.0f)\n", dir.x, dir.y, dir.z);
		cosine_pdf pdf(dir);
		printf("  u:(%.0f, %.0f, %.0f)\n", pdf.uvw.u().x, pdf.uvw.u().y, pdf.uvw.u().z);
		printf("  v:(%.0f, %.0f, %.0f)\n", pdf.uvw.v().x, pdf.uvw.v().y, pdf.uvw.v().z);
		printf("  w:(%.0f, %.0f, %.0f)\n", pdf.uvw.w().x, pdf.uvw.w().y, pdf.uvw.w().z);
	}
    return 0;
}

