#define Capacity 1024
#define M_PI 3.14159265358979323846f

struct HitInfo {
	int hit;
	float t;
	float hitPoint[3];
	float hitNormal[3];
	float color[3];
	float roughness;
	float metallic;
	float emission;
	int volumeIdx;
	int triangleIdx;
};

struct Volume {
	float BBoxMin[3];
	float BBoxMax[3];
	int count;
	float v1[Capacity * 3];
	float v2[Capacity * 3];
	float v3[Capacity * 3];
	float Roughness[Capacity];
	float Metallic[Capacity];
	float Emission[Capacity];
	float normals[Capacity * 3];
	float colors[Capacity * 3];
};

struct Cluster {
	float BBoxMin[3];
	float BBoxMax[3];
	struct Volume volumes[8];
};

struct Block {
	float BBoxMin[3];
	float BBoxMax[3];
	struct Cluster clusters[8];
};

struct Region {
	float BBoxMin[3];
	float BBoxMax[3];
	struct Block blocks[8];
};

int rayBoxIntersect(float rayOrigin[3], float rayDir[3],
					float boxMin[3], float boxMax[3],
					float *tMin, float *tMax) {
	const float epsilon = 1e-6f;
	float t1 = -1e30f;
	float t2 = 1e30f;

	for (int i = 0; i < 3; i++) {
		if (fabsf(rayDir[i]) < epsilon) {
			if (rayOrigin[i] < boxMin[i] || rayOrigin[i] > boxMax[i]) {
				return 0;
			}
		} else {
			float invD = 1.0f / rayDir[i];
			float t_near = (boxMin[i] - rayOrigin[i]) * invD;
			float t_far = (boxMax[i] - rayOrigin[i]) * invD;

			if (t_near > t_far) {
				float temp = t_near;
				t_near = t_far;
				t_far = temp;
			}

			t1 = (t_near > t1) ? t_near : t1;
			t2 = (t_far < t2) ? t_far : t2;

			if (t1 > t2) {
				return 0;
			}
		}
	}

	*tMin = t1;
	*tMax = t2;
	return (t2 >= 0.0f);
}

int rayTriangleIntersect(float rayOrigin[3], float rayDir[3],
						 float v1[3], float v2[3], float v3[3],
						 float *t, float hitNormal[3]) {
	const float epsilon = 1e-6f;

	float edge1[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
	float edge2[3] = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};

	float h[3] = {
		rayDir[1] * edge2[2] - rayDir[2] * edge2[1],
		rayDir[2] * edge2[0] - rayDir[0] * edge2[2],
		rayDir[0] * edge2[1] - rayDir[1] * edge2[0]};

	float a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

	if (a > -epsilon && a < epsilon) {
		return 0;
	}

	float f = 1.0f / a;
	float s[3] = {rayOrigin[0] - v1[0], rayOrigin[1] - v1[1], rayOrigin[2] - v1[2]};
	float u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

	if (u < 0.0f || u > 1.0f) {
		return 0;
	}

	float q[3] = {
		s[1] * edge1[2] - s[2] * edge1[1],
		s[2] * edge1[0] - s[0] * edge1[2],
		s[0] * edge1[1] - s[1] * edge1[0]};

	float v = f * (rayDir[0] * q[0] + rayDir[1] * q[1] + rayDir[2] * q[2]);

	if (v < 0.0f || u + v > 1.0f) {
		return 0;
	}

	float tValue = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);

	if (tValue > epsilon) {
		*t = tValue;

		float normal[3] = {
			edge1[1] * edge2[2] - edge1[2] * edge2[1],
			edge1[2] * edge2[0] - edge1[0] * edge2[2],
			edge1[0] * edge2[1] - edge1[1] * edge2[0]};

		float len = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
		if (len > epsilon) {
			hitNormal[0] = normal[0] / len;
			hitNormal[1] = normal[1] / len;
			hitNormal[2] = normal[2] / len;
		}

		return 1;
	}

	return 0;
}

float3 sampleSkybox(
	const float3 rayDir,
	__global const unsigned char *SkyBoxTop,
	__global const unsigned char *SkyBoxBottom,
	__global const unsigned char *SkyBoxLeft,
	__global const unsigned char *SkyBoxRight,
	__global const unsigned char *SkyBoxFront,
	__global const unsigned char *SkyBoxBack,
	const int skyBoxWidth,
	const int skyBoxHeight) {
	
	float3 dir = normalize(rayDir);
	float3 absDir = fabs(dir);
	float maxComponent = max(max(absDir.x, absDir.y), absDir.z);
	const float epsilon = 1e-6f;

	float2 uv;
	__global const unsigned char *selectedFace = NULL;

	if (maxComponent == absDir.x && fabs(dir.x) > epsilon) {
		if (dir.x > 0) {
			uv.x = (-dir.z / dir.x + 1.0f) * 0.5f;
			uv.y = (-dir.y / dir.x + 1.0f) * 0.5f;
			selectedFace = SkyBoxRight;
		} else {
			uv.x = (dir.z / (-dir.x) + 1.0f) * 0.5f;
			uv.y = (-dir.y / (-dir.x) + 1.0f) * 0.5f;
			selectedFace = SkyBoxLeft;
		}
	} else if (maxComponent == absDir.y && fabs(dir.y) > epsilon) {
		if (dir.y > 0) {
			uv.x = (dir.x / dir.y + 1.0f) * 0.5f;
			uv.y = (dir.z / dir.y + 1.0f) * 0.5f;
			selectedFace = SkyBoxTop;
		} else {
			uv.x = (dir.x / (-dir.y) + 1.0f) * 0.5f;
			uv.y = (-dir.z / (-dir.y) + 1.0f) * 0.5f;
			selectedFace = SkyBoxBottom;
		}
	} else if (fabs(dir.z) > epsilon) {
		if (dir.z > 0) {
			uv.x = (dir.x / dir.z + 1.0f) * 0.5f;
			uv.y = (-dir.y / dir.z + 1.0f) * 0.5f;
			selectedFace = SkyBoxFront;
		} else {
			uv.x = (-dir.x / (-dir.z) + 1.0f) * 0.5f;
			uv.y = (-dir.y / (-dir.z) + 1.0f) * 0.5f;
			selectedFace = SkyBoxBack;
		}
	}

	uv = clamp(uv, 0.0f, 1.0f);
	float3 skyboxColor = (float3)(0.5f, 0.7f, 1.0f);

	if (selectedFace != NULL && skyBoxWidth > 0 && skyBoxHeight > 0) {
		int texX = clamp((int)(uv.x * (skyBoxWidth - 1)), 0, skyBoxWidth - 1);
		int texY = clamp((int)(uv.y * (skyBoxHeight - 1)), 0, skyBoxHeight - 1);
		int texIndex = (texY * skyBoxWidth + texX) * 3;

		skyboxColor.x = selectedFace[texIndex] / 255.0f;
		skyboxColor.y = selectedFace[texIndex + 1] / 255.0f;
		skyboxColor.z = selectedFace[texIndex + 2] / 255.0f;
	}

	return skyboxColor;
}

struct HitInfo intersectRay(__global const struct Region *region, float3 pos, float3 dir) {
	struct HitInfo result;
	result.hit = 0;
	result.t = 1e30f;

	const float epsilon = 1e-6f;
	const float inf = 1e30f;

	float posArr[3] = {pos.x, pos.y, pos.z};
	float dirArr[3] = {dir.x, dir.y, dir.z};
	
	float tMin, tMax;
	if (!rayBoxIntersect(posArr, dirArr, region->BBoxMin, region->BBoxMax, &tMin, &tMax)) {
		return result;
	}

	for (int b = 0; b < 8; b++) {
		__global const struct Block *block = &region->blocks[b];

		if (block->BBoxMin[0] >= inf || block->BBoxMin[1] >= inf || block->BBoxMin[2] >= inf) {
			continue;
		}

		if (!rayBoxIntersect(posArr, dirArr, block->BBoxMin, block->BBoxMax, &tMin, &tMax)) {
			continue;
		}

		for (int c = 0; c < 8; c++) {
			__global const struct Cluster *cluster = &block->clusters[c];

			if (cluster->BBoxMin[0] >= inf || cluster->BBoxMin[1] >= inf || cluster->BBoxMin[2] >= inf) {
				continue;
			}

			if (!rayBoxIntersect(posArr, dirArr, cluster->BBoxMin, cluster->BBoxMax, &tMin, &tMax)) {
				continue;
			}

			for (int v = 0; v < 8; v++) {
				__global const struct Volume *volume = &cluster->volumes[v];

				if (volume->count == 0) {
					continue;
				}

				if (volume->BBoxMin[0] >= inf || volume->BBoxMin[1] >= inf || volume->BBoxMin[2] >= inf) {
					continue;
				}

				if (!rayBoxIntersect(posArr, dirArr, volume->BBoxMin, volume->BBoxMax, &tMin, &tMax)) {
					continue;
				}

				for (int i = 0; i < volume->count; i++) {
					float v1[3] = {volume->v1[i * 3], volume->v1[i * 3 + 1], volume->v1[i * 3 + 2]};
					float v2[3] = {volume->v2[i * 3], volume->v2[i * 3 + 1], volume->v2[i * 3 + 2]};
					float v3[3] = {volume->v3[i * 3], volume->v3[i * 3 + 1], volume->v3[i * 3 + 2]};

					float t = 0.0f;
					float normal[3] = {0, 0, 0};

					if (rayTriangleIntersect(posArr, dirArr, v1, v2, v3, &t, normal)) {
						if (t > epsilon && t < result.t) {
							result.hit = 1;
							result.t = t;

							result.hitPoint[0] = pos.x + dir.x * t;
							result.hitPoint[1] = pos.y + dir.y * t;
							result.hitPoint[2] = pos.z + dir.z * t;

							result.hitNormal[0] = normal[0];
							result.hitNormal[1] = normal[1];
							result.hitNormal[2] = normal[2];

							result.color[0] = volume->colors[i * 3] / 255.0f;
							result.color[1] = volume->colors[i * 3 + 1] / 255.0f;
							result.color[2] = volume->colors[i * 3 + 2] / 255.0f;

							result.roughness = volume->Roughness[i];
							result.metallic = volume->Metallic[i];
							result.emission = volume->Emission[i];

							result.volumeIdx = v;
							result.triangleIdx = i;
						}
					}
				}
			}
		}
	}

	return result;
}

int intersectAny(__global const struct Region *region, float3 pos, float3 dir, float maxDist) {
	const float epsilon = 1e-6f;
	const float inf = 1e30f;

	float posArr[3] = {pos.x, pos.y, pos.z};
	float dirArr[3] = {dir.x, dir.y, dir.z};
	
	float tMin, tMax;
	if (!rayBoxIntersect(posArr, dirArr, region->BBoxMin, region->BBoxMax, &tMin, &tMax)) {
		return 0;
	}

	for (int b = 0; b < 8; b++) {
		__global const struct Block *block = &region->blocks[b];

		if (block->BBoxMin[0] >= inf || block->BBoxMin[1] >= inf || block->BBoxMin[2] >= inf) {
			continue;
		}

		if (!rayBoxIntersect(posArr, dirArr, block->BBoxMin, block->BBoxMax, &tMin, &tMax)) {
			continue;
		}

		for (int c = 0; c < 8; c++) {
			__global const struct Cluster *cluster = &block->clusters[c];

			if (cluster->BBoxMin[0] >= inf || cluster->BBoxMin[1] >= inf || cluster->BBoxMin[2] >= inf) {
				continue;
			}

			if (!rayBoxIntersect(posArr, dirArr, cluster->BBoxMin, cluster->BBoxMax, &tMin, &tMax)) {
				continue;
			}

			for (int v = 0; v < 8; v++) {
				__global const struct Volume *volume = &cluster->volumes[v];

				if (volume->count == 0) {
					continue;
				}

				if (volume->BBoxMin[0] >= inf || volume->BBoxMin[1] >= inf || volume->BBoxMin[2] >= inf) {
					continue;
				}

				if (!rayBoxIntersect(posArr, dirArr, volume->BBoxMin, volume->BBoxMax, &tMin, &tMax)) {
					continue;
				}

				for (int i = 0; i < volume->count; i++) {
					float v1[3] = {volume->v1[i * 3], volume->v1[i * 3 + 1], volume->v1[i * 3 + 2]};
					float v2[3] = {volume->v2[i * 3], volume->v2[i * 3 + 1], volume->v2[i * 3 + 2]};
					float v3[3] = {volume->v3[i * 3], volume->v3[i * 3 + 1], volume->v3[i * 3 + 2]};

					float t = 0.0f;
					float normal[3] = {0, 0, 0};

					if (rayTriangleIntersect(posArr, dirArr, v1, v2, v3, &t, normal)) {
						if (t > epsilon && t <= maxDist) {
							return 1;
						}
					}
				}
			}
		}
	}

	return 0;
}

float fract(float x) {
	return x - floor(x);
}

float hash(float seed) {
	return fract(sin(seed) * 43758.5453f);
}

float2 rand2(float3 rayOrigin, float3 incident, int bounce) {
	float seed = dot(rayOrigin, (float3)(12.9898f, 78.233f, 37.719f)) +
				 dot(incident, (float3)(39.346f, 11.135f, 83.155f)) +
				 (float)bounce * 17.0f;
	float rand1 = hash(seed);
	float rand2 = hash(seed + 1.0f);
	return (float2)(rand1, rand2);
}

float3 cosineWeightedHemisphere(float3 normal, float2 random) {
	float r1 = random.x;
	float r2 = random.y;
	
	float phi = 2.0f * M_PI * r1;
	float cosTheta = sqrt(r2);
	float sinTheta = sqrt(1.0f - r2);
	
	float x = cos(phi) * sinTheta;
	float y = sin(phi) * sinTheta;
	float z = cosTheta;
	
	float3 tangent, bitangent;
	if (fabs(normal.x) > fabs(normal.y)) {
		tangent = normalize((float3)(normal.z, 0, -normal.x));
	} else {
		tangent = normalize((float3)(0, normal.z, -normal.y));
	}
	bitangent = cross(normal, tangent);
	
	return normalize(x * tangent + y * bitangent + z * normal);
}

float3 ggxSample(float3 normal, float roughness, float2 random) {
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	
	float phi = 2.0f * M_PI * random.x;
	float cosTheta = sqrt((1.0f - random.y) / (1.0f + (alpha2 - 1.0f) * random.y));
	float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
	
	float x = cos(phi) * sinTheta;
	float y = sin(phi) * sinTheta;
	float z = cosTheta;
	
	float3 tangent, bitangent;
	if (fabs(normal.x) > fabs(normal.y)) {
		tangent = normalize((float3)(normal.z, 0, -normal.x));
	} else {
		tangent = normalize((float3)(0, normal.z, -normal.y));
	}
	bitangent = cross(normal, tangent);
	
	return normalize(x * tangent + y * bitangent + z * normal);
}

float3 fresnelSchlick(float cosTheta, float3 F0) {
	return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

float ggxDistribution(float3 normal, float3 halfVec, float roughness) {
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float NdotH = max(dot(normal, halfVec), 0.0f);
	float NdotH2 = NdotH * NdotH;
	
	float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
	denom = M_PI * denom * denom;
	
	return alpha2 / max(denom, 0.0001f);
}

float geometrySchlickGGX(float NdotV, float roughness) {
	float r = roughness + 1.0f;
	float k = (r * r) / 8.0f;
	
	return NdotV / (NdotV * (1.0f - k) + k);
}

float geometrySmith(float3 normal, float3 viewDir, float3 lightDir, float roughness) {
	float NdotV = max(dot(normal, viewDir), 0.0f);
	float NdotL = max(dot(normal, lightDir), 0.0f);
	float ggx1 = geometrySchlickGGX(NdotV, roughness);
	float ggx2 = geometrySchlickGGX(NdotL, roughness);
	
	return ggx1 * ggx2;
}

float3 evaluateBRDF(float3 viewDir, float3 lightDir, float3 normal, 
					float3 albedo, float roughness, float metallic) {
	float3 halfVec = normalize(viewDir + lightDir);
	
	float3 F0 = mix((float3)(0.04f), albedo, metallic);
	float3 F = fresnelSchlick(max(dot(halfVec, viewDir), 0.0f), F0);
	
	float D = ggxDistribution(normal, halfVec, roughness);
	float G = geometrySmith(normal, viewDir, lightDir, roughness);
	
	float3 specular = (D * G * F) / max(4.0f * max(dot(normal, viewDir), 0.0f) * max(dot(normal, lightDir), 0.0f), 0.0001f);
	
	float3 kS = F;
	float3 kD = (1.0f - kS) * (1.0f - metallic);
	
	float3 diffuse = kD * albedo / M_PI;
	
	return diffuse + specular;
}

__kernel void rayTraceScene(
	__global const struct Region *sceneRegion,
	__global float *ScreenDistances,
	__global float *ScreenNormals,
	__global float *ScreenColors,
	const float3 camPos,
	const float3 camDir,
	const float fov,
	const int screenWidth,
	const int screenHeight,
	__global const unsigned char *SkyBoxTop,
	__global const unsigned char *SkyBoxBottom,
	__global const unsigned char *SkyBoxLeft,
	__global const unsigned char *SkyBoxRight,
	__global const unsigned char *SkyBoxFront,
	__global const unsigned char *SkyBoxBack,
	const int skyBoxWidth,
	const int skyBoxHeight,
	const float3 sunDir,
	const float3 sunColor,
	const float sunIntensity,
	const int maxBounces) {
	
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x >= screenWidth || y >= screenHeight) return;

	int pixelIndex = y * screenWidth + x;

	float3 forward = normalize(camDir);
	float3 camUp = (float3)(0.0f, 1.0f, 0.0f);
	float3 right = normalize(cross(forward, camUp));
	float3 up = cross(right, forward);

	float ndcX = (x + 0.5f) / screenWidth * 2.0f - 1.0f;
	float ndcY = -((y + 0.5f) / screenHeight * 2.0f - 1.0f);

	float3 rayDir = normalize(forward + ndcX * right * fov + ndcY * up * fov);

	float3 radiance = (float3)(0.0f);
	float3 throughput = (float3)(1.0f);
	float3 currentPos = camPos;
	float3 currentDir = rayDir;

	for (int bounce = 0; bounce < maxBounces; bounce++) {
		struct HitInfo hit = intersectRay(sceneRegion, currentPos, currentDir);

		if (!hit.hit) {
			float3 skyColor = sampleSkybox(currentDir, SkyBoxTop, SkyBoxBottom, SkyBoxLeft, 
										   SkyBoxRight, SkyBoxFront, SkyBoxBack, skyBoxWidth, skyBoxHeight);
			radiance += throughput * skyColor;
			break;
		}

		float3 hitPos = (float3)(hit.hitPoint[0], hit.hitPoint[1], hit.hitPoint[2]);
		float3 normal = normalize((float3)(hit.hitNormal[0], hit.hitNormal[1], hit.hitNormal[2]));
		float3 albedo = (float3)(hit.color[0], hit.color[1], hit.color[2]);

		radiance += throughput * albedo * hit.emission;

		float3 shadowRayOrigin = hitPos + normal * 0.001f;
		float3 shadowRayDir = normalize(sunDir);
		int inShadow = intersectAny(sceneRegion, shadowRayOrigin, shadowRayDir, 10000.0f);

		if (!inShadow) {
			float NdotL = max(dot(normal, shadowRayDir), 0.0f);
			float3 viewDir = normalize(camPos - hitPos);
			float3 brdf = evaluateBRDF(viewDir, shadowRayDir, normal, albedo, hit.roughness, hit.metallic);
			radiance += throughput * sunColor * sunIntensity * brdf * NdotL;
		}

		float2 random = rand2(hitPos, currentDir, bounce);
		
		float3 newDir;
		if (hit.roughness < 0.1f && hit.metallic > 0.5f) {
			float3 reflectDir = reflect(currentDir, normal);
			float3 H = ggxSample(normal, hit.roughness, random);
			newDir = normalize(reflect(-currentDir, H));
			if (dot(newDir, normal) < 0.0f) {
				newDir = reflectDir;
			}
		} else {
			newDir = cosineWeightedHemisphere(normal, random);
		}

		float3 viewDir = -currentDir;
		float3 brdf = evaluateBRDF(viewDir, newDir, normal, albedo, hit.roughness, hit.metallic);
		float NdotL = max(dot(normal, newDir), 0.0f);
		float pdf = NdotL / M_PI;

		throughput *= brdf * NdotL / max(pdf, 0.0001f);

		if (length(throughput) < 0.001f) {
			break;
		}

		currentPos = hitPos + newDir * 0.001f;
		currentDir = newDir;
	}

	int colorIndex = pixelIndex * 3;
	ScreenColors[colorIndex] = clamp(radiance.x, 0.0f, 1.0f);
	ScreenColors[colorIndex + 1] = clamp(radiance.y, 0.0f, 1.0f);
	ScreenColors[colorIndex + 2] = clamp(radiance.z, 0.0f, 1.0f);

	ScreenDistances[pixelIndex] = 0.0f;
	ScreenNormals[pixelIndex * 3] = 0.0f;
	ScreenNormals[pixelIndex * 3 + 1] = 0.0f;
	ScreenNormals[pixelIndex * 3 + 2] = 0.0f;
}
			