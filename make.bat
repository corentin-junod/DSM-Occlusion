cls
cmake -B ./build
cmake --build ./build --config Release
set PATH=%PATH%;C:\OSGeo4W\bin
.\build\Release\DSM_Occlusion.exe