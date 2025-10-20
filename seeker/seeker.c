#include <math.h>
#include <float.h>
#include "../fireSim/fireSim.h"

static float dot3(const float a[3], const float b[3]) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void normalize3(float v[3]) {
	float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (len > 0.0f) {
		v[0] /= len;
		v[1] /= len;
		v[2] /= len;
	}
}

// Returns index of hottest object in seeker FOV
// centerWeight: 0.0-1.0, how much center alignment matters
// tempWeight: 0.0-1.0, how much temperature matters
// distWeight: 0.0-1.0, how much closeness matters
// (weights should sum to 1.0 for best results)
int findClosestObjectToViewCenter(
    const float pos[3],
    float dir[3],
    float FOV,
    const float *ObjX,
    const float *ObjY,
    const float *ObjZ,
    const float *tempObj,
    const float *seekerImageDistances,
    int numOfObj,
    float centerWeight,
    float tempWeight,
    float distWeight
) {
    if (numOfObj <= 0) return -1;
    
    float viewDir[3] = {dir[0], dir[1], dir[2]};
    normalize3(viewDir);
    
    float fovHalfRad = (FOV / 2.0f) * (M_PI / 180.0f);
    float fovCosThreshold = cosf(fovHalfRad);
    
    int bestIdx = -1;
    float bestScore = -FLT_MAX;
    
    // Find max temp for normalization
    float maxTemp = 0.0f;
    for (int i = 0; i < numOfObj; i++) {
        if (tempObj[i] > maxTemp) maxTemp = tempObj[i];
    }
    if (maxTemp < 0.001f) maxTemp = 1.0f;
    
    for (int i = 0; i < numOfObj; i++) {
        float objPos[3] = {ObjX[i], ObjY[i], ObjZ[i]};
        float toObj[3] = {
            objPos[0] - pos[0],
            objPos[1] - pos[1],
            objPos[2] - pos[2]
        };
        
        float dist = sqrtf(toObj[0] * toObj[0] + toObj[1] * toObj[1] + toObj[2] * toObj[2]);
        
        if (dist < 0.001f) continue;
        
        float toObjNorm[3] = {toObj[0] / dist, toObj[1] / dist, toObj[2] / dist};
        float cosAngle = dot3(viewDir, toObjNorm);
        
        if (cosAngle < fovCosThreshold) continue;
        
        // Check if object is occluded by geometry using seeker depth buffer
        // Map object to seeker image coordinates
        float rightVec[3], upVec[3];
        // Create coordinate system from view direction
        if (fabsf(viewDir[1]) < 0.99f) {
            rightVec[0] = viewDir[2];
            rightVec[1] = 0.0f;
            rightVec[2] = -viewDir[0];
        } else {
            rightVec[0] = 1.0f;
            rightVec[1] = 0.0f;
            rightVec[2] = 0.0f;
        }
        normalize3(rightVec);
        upVec[0] = rightVec[1] * viewDir[2] - rightVec[2] * viewDir[1];
        upVec[1] = rightVec[2] * viewDir[0] - rightVec[0] * viewDir[2];
        upVec[2] = rightVec[0] * viewDir[1] - rightVec[1] * viewDir[0];
        
        float xProj = dot3(toObjNorm, rightVec);
        float yProj = dot3(toObjNorm, upVec);
        
        float angle = acosf(fmaxf(-1.0f, fminf(1.0f, cosAngle)));
        float pixelX = (xProj / tanf(fovHalfRad)) * 0.5f + 0.5f;
        float pixelY = (yProj / tanf(fovHalfRad)) * 0.5f + 0.5f;
        
        int px = (int)(pixelX * MISSILE_SEEKER_SIZE);
        int py = (int)(pixelY * MISSILE_SEEKER_SIZE);
        
        // Check if object is occluded
        if (px >= 0 && px < MISSILE_SEEKER_SIZE && py >= 0 && py < SEEKER_SIZE) {
            int pixelIdx = py * MISSILE_SEEKER_SIZE + px;
            float geometryDist = seekerImageDistances[pixelIdx];
            
            // Object is occluded if geometry is closer (with small tolerance)
            if (geometryDist > 0.0f && geometryDist < dist - 0.1f) {
                continue;
            }
        }
        
        // Angular score: 1.0 (centered) to 0.0 (edge of FOV)
        float angularScore = 1.0f - (angle / fovHalfRad);
        
        // Temperature with inverse square falloff (like real IR radiation)
        float tempAtSeeker = (tempObj[i] / maxTemp) / (dist * dist);
        float tempScore = tempAtSeeker;
        
        // Distance score: inverse so closer is better
        float distScore = 1.0f / (1.0f + dist);
        
        // Combined score (higher is better)
        float score = centerWeight * angularScore + 
                     tempWeight * tempScore + 
                     distWeight * distScore;
        
        if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }
    
    return bestIdx;
}