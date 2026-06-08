// LegoVision - Physics Belt Integration Patch
// Sobrescribe startBeltSimulation para usar physBelt cuando hay sesion activa
// y aniade el slider de densidad al DOM de Vista en Vivo

document.addEventListener("DOMContentLoaded",function(){
  initPhysBelt();
  initPhysBeltHistoryNav();
  removeCaptureFrameButton();
  patchStartBeltSimulation();
  patchInitSessionControls();
});

function removeCaptureFrameButton(){
  var btn=document.getElementById("btn-capture-frame");
  if(btn)btn.style.display="none";
}

function patchStartBeltSimulation(){
  var origStart=window.startBeltSimulation;
  window.startBeltSimulation=function(skipInit){
    // Si hay render de Blender activo (setSimulationActive), usar el loop estandar de animationLoop
    // que ya maneja el scroll de la imagen. No activar physBelt en ese caso.
    if(sessionActive && physBeltActive===false && !setSimulationActive){
      if(window.cancelAnimationFrame&&window.simulationRafId){cancelAnimationFrame(window.simulationRafId);window.simulationRafId=null;}
      activatePhysBelt();
      window.simulationRafId=requestAnimationFrame(physBeltAnimLoop);
    }else{
      if(origStart)origStart(skipInit);
    }
  };
  var origStop=window.stopBeltSimulation;
  window.stopBeltSimulation=function(){
    if(physBeltActive){deactivatePhysBelt();}
    if(origStop)origStop();
  };
}

function patchInitSessionControls(){
  // En el nuevo flujo inferencia-test:
  // Si hay render Blender disponible (setSimulationActive=true), NO activar physBelt.
  // Solo activar physBelt como fallback si no hay imagen de Blender.
  var btn=document.getElementById("btn-toggle-session");
  if(!btn)return;
  var obs=new MutationObserver(function(){
    if(sessionActive && !physBeltActive && !setSimulationActive){
      // Solo activar physBelt si NO hay render Blender activo
      initPhysBelt();
      activatePhysBelt();
    }else if(!sessionActive&&physBeltActive){
      deactivatePhysBelt();
    }
  });
  obs.observe(btn,{attributes:true,childList:false});
}

var physBeltRafId=null;
var lastPhysFrameTime=0;
function physBeltAnimLoop(timestamp){
  if(!physBeltActive){
    lastPhysFrameTime=0;
    return;
  }
  physBeltRafId=requestAnimationFrame(physBeltAnimLoop);
  simulationRafId=physBeltRafId;
  if(!lastPhysFrameTime)lastPhysFrameTime=timestamp;
  var dt=(timestamp-lastPhysFrameTime)/1000;
  lastPhysFrameTime=timestamp;
  if(dt>0.1)dt=0.1;
  animatePhysBelt(timestamp,dt);
}
